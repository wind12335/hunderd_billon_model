#!/usr/bin/env bash
# =============================================================================
# 双机 16 卡多机链路测试启动脚本（两种平台自适应）
#
# 【SCNet 模型训练模块 / K8s vcjob 模式】← 平台用 /bin/sh 在两个 worker 上
#   各执行一次本脚本，无需任何参数：
#     * pod 名以 -worker-0 结尾 → 主节点：体检→初始化子模块→生成脚本→
#       把本机 IP 写入共享目录 → 以 NODE_RANK=0 启动训练
#     * pod 名以 -worker-1 结尾 → 从节点：等待主节点就绪 → 以 NODE_RANK=1 启动
#   依赖：/root/private_data/thousand_model 为两 worker 共享挂载。
#
# 【裸机 SSH 模式】（手动在 worker0 上执行）
#   bash scripts/run_dual_node_test.sh <worker1地址> [--mode check|generate|smoke|run|stop]
#   选项: --worker0 <addr> --model-path <dir> --master-port <p> --train-iters <n>
#         --seq-len <n> --skip-checks
#
# 说明：数据走 mock（--mock-data），无需准备数据集；随机初始化，loss 数值无意义，
#       验证目标是 rendezvous/RCCL/编排/Megatron 启动链路。
# 通信：跨节点默认自动启用 shca RDMA（资产在共享存储 second_hygon/rdma-shca，
#       每作业离线装一次用户态 ~10s，失败自动回退 TCP）；RDMA_DISABLE=1 可关闭。
# =============================================================================
# --- POSIX 前奏：平台可能用 /bin/sh 调起，先切换到 bash ---
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
# 心跳行：平台日志若连这一行都没有，说明脚本根本没被执行（pod/暂存问题），
# 与脚本内容无关——这是区分"脚本问题"和"平台问题"的分界线
printf '[dual-test] 脚本已启动 hostname=%s bash=%s\n' "$(hostname 2>/dev/null || echo '?')" "${BASH_VERSION}"
set -euo pipefail
# 任何命令失败都在失败现场把行号+退出码打到 stdout（平台日志不收 stderr），
# 杜绝"静默死"——上一版 worker-0 死在 ip 探测行、退出码 127、零输出，就是这类问题
trap 'rc=$?; printf "\033[1;31m[dual-test][异常退出]\033[0m 脚本在第 %s 行以退出码 %s 终止（把此行号连同前后日志发我）\n" "$LINENO" "$rc"' ERR

# ── 默认参数 ─────────────────────────────────────────────────────────────────
WORKER1=""
WORKER0=""
MODEL_PATH="model/Llama-3.1-8B"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ITERS=20
SEQ_LEN=4096
MODE="auto"
SKIP_CHECKS=0
GPUS_PER_NODE=8
NUM_NODES=2

REPO_ROOT=""   # 由 detect_repo_root() 解析：平台可能把启动脚本 staging 到 / 执行，不能依赖脚本自身路径
OUT_DIR="${REPO_ROOT}/generated/dual16_llama8b"
PLAN_SCRIPT="${OUT_DIR}/launch_megatron_plan.sh"
ADDR_MARKER="${OUT_DIR}/.vcjob_master_addr"

log()  { printf '\033[1;32m[dual-test]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dual-test][警告]\033[0m %s\n' "$*"; }
# 失败信息同时写 stdout+stderr：平台日志收集疑似不收 stderr，只写 stderr
# 的 die() 会表现为"零日志静默死"（2026-10-07 worker-0 实际发生）
die()  { printf '\033[1;31m[dual-test][失败]\033[0m %s\n' "$*" | tee /dev/stderr; exit 1; }

usage() {
    cat <<'EOF'
用法:
  SCNet vcjob（平台自动双 worker 执行）: 直接把本脚本填为启动脚本，无需参数
  裸机: bash scripts/run_dual_node_test.sh <worker1地址> [选项]
选项:
  --worker0 <addr>     手动指定本机/主节点地址（默认自动侦测）
  --model-path <dir>   模型目录（默认 model/Llama-3.1-8B）
  --master-port <p>    rendezvous 端口（默认 29500）
  --train-iters <n>    训练步数（默认 20）
  --seq-len <n>        序列长度（默认 4096）
  --mode <m>           auto|check|generate|smoke|run|stop（默认 auto）
  --skip-checks        跳过体检（不建议）
EOF
}

# ── 参数解析 ─────────────────────────────────────────────────────────────────
 positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --worker0)     WORKER0="$2"; shift 2 ;;
        --model-path)  MODEL_PATH="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --train-iters) TRAIN_ITERS="$2"; shift 2 ;;
        --seq-len)     SEQ_LEN="$2"; shift 2 ;;
        --mode)        MODE="$2"; shift 2 ;;
        --skip-checks) SKIP_CHECKS=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             positional+=("$1"); shift ;;
    esac
done
[[ ${#positional[@]} -gt 0 ]] && WORKER1="${positional[0]}"

# ── 仓库根探测：依次尝试环境变量/脚本所在目录/常见挂载点 ────────────────────
# 共享存储挂载偶发晚于容器启动（本平台 run-2/本次均出现过 worker-0 起跑时
# 未挂好 → detect 直接 die → 零日志），故加 120 秒重试窗口
detect_repo_root() {
    local script_dir; script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    local candidates=(
        "${THOUSAND_MODEL_ROOT:-}"
        "${script_dir}/.."
        "${script_dir}"
        "/root/private_data/thousand_model"
        "${HOME:-/root}/private_data/thousand_model"
        "/root/private_data"
        "/mnt/data/thousand_model"
        "/mnt/data"
    )
    local c waited=0
    while :; do
        for c in "${candidates[@]}"; do
            [[ -z "$c" ]] && continue
            c="$(cd "$c" 2>/dev/null && pwd)" || continue
            if [[ -f "$c/tools/launch_cluster.py" && -f "$c/megatron_mvp/search.py" ]]; then
                REPO_ROOT="$c"
                OUT_DIR="${REPO_ROOT}/generated/dual16_llama8b"
                PLAN_SCRIPT="${OUT_DIR}/launch_megatron_plan.sh"
                ADDR_MARKER="${OUT_DIR}/.vcjob_master_addr"
                return
            fi
        done
        [[ $waited -ge 120 ]] && break
        if [[ $waited -eq 0 ]]; then
            log "共享存储暂不可见（挂载可能未就绪），每 10 秒重试，最多 120 秒..."
        fi
        sleep 10; waited=$((waited+10))
    done
    die "找不到仓库根（外挂目录未挂载到预期路径？重试 120 秒后放弃）。已尝试: ${candidates[*]}；可用环境变量 THOUSAND_MODEL_ROOT=<仓库根> 指定"
}

# ── 平台侦测：SCNet vcjob（pod 名 *-worker-N）───────────────────────────────
# 同一 vcjob 的两个 pod 共享 -worker-N 之前的前缀（JOB_TAG）。地址标记文件
# 必须带 JOB_TAG 命名：共享存储跨作业持久，无标记的旧文件会让新作业的
# worker-1 读到上一作业已销毁 pod 的 IP，然后去等一个永远不会开的端口
# （2026-10-07 实际踩中：worker-1 等 172.31.50.196，真主节点是 .207）。
ROLE=""
JOB_TAG=""
is_vcjob() {
    local h="${HOSTNAME:-$(hostname)}"
    [[ "$h" =~ ^(.*)-worker-([0-9]+)$ ]] || return 1
    JOB_TAG="${BASH_REMATCH[1]}"
    ROLE="${BASH_REMATCH[2]}"
    ADDR_MARKER="${OUT_DIR}/.vcjob_master_addr.${JOB_TAG}"
    return 0
}

detect_local_ip() {
    local ip
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "$ip" ]] || die "无法侦测本机 IP（hostname -I 为空）"
    echo "$ip"
}

# ── 本地体检（两种模式共用的本机检查）───────────────────────────────────────
local_preflight() {
    local fail=0
    log "── 本机 python3/torch"
    if command -v python3 >/dev/null 2>&1 && python3 -c 'import torch' 2>/dev/null; then
        log "  ✓ $(python3 -c 'import torch; print(torch.__version__, "cards:", torch.cuda.device_count())')"
    else
        warn "  ✗ python3/torch 不可用"; fail=1
    fi
    log "── 本机模型目录 ${REPO_ROOT}/${MODEL_PATH}"
    [[ -d "${REPO_ROOT}/${MODEL_PATH}" ]] && log "  ✓ 在位" || { warn "  ✗ 缺失（注意脚本实际检查的绝对路径）"; fail=1; }
    [[ $fail -eq 0 ]] || die "本机体检未通过"
}

ensure_submodules() {
    local f="${REPO_ROOT}/Megatron-LM-das/3rdparty/Megatron-LM/megatron/training/arguments.py"
    if [[ -f "$f" ]]; then log "── Megatron 子模块 ✓ 已初始化"; return; fi
    log "── Megatron 子模块为空，尝试自动初始化（需访问 GitHub）..."
    (cd "${REPO_ROOT}/Megatron-LM-das" && git submodule update --init --recursive) \
        || die "子模块初始化失败：请检查网络/GitHub 可达性，或手动在可联网环境完成后再提交作业"
    [[ -f "$f" ]] || die "子模块初始化后仍未找到 ${f}"
    log "  ✓ 子模块初始化成功"
}

# ── 规划 + 生成（world=16 硬校验，防拓扑错配 bug）───────────────────────────
generate() {
    log "生成多机脚本（预期 world_size=16）..."
    cd "$REPO_ROOT"; export PYTHONPATH="$REPO_ROOT"
    local gen_log; gen_log="$(mktemp /tmp/dual16_gen.XXXXXX.log)"
    python3 examples/autoparallel_multinode_train.py \
        --num-nodes "$NUM_NODES" --gpus-per-node "$GPUS_PER_NODE" \
        --model-path "$MODEL_PATH" \
        --backend MEGATRON --run-mode LOWER \
        --sequence-length "$SEQ_LEN" --train-iters "$TRAIN_ITERS" \
        --output-dir "$OUT_DIR" 2>&1 | tee "$gen_log"
    grep -q "world_size=16" "$gen_log" || die "world_size != 16（拓扑错配 bug 或显存判定异常），已阻止启动。日志: $gen_log"
    log "world_size=16 校验通过"
    [[ -f "$PLAN_SCRIPT" ]] || die "未找到生成脚本 ${PLAN_SCRIPT}"
}

# ── shca RDMA：跨节点通信从 NET/Socket(TCP) 升级为 RDMA ──────────────────────
# 平台配方（与 second_hygon phase4 实测版一致，NET/IBext_v8 已验证，shca_0..3
# 为 400G NDR）。两个要点：
#  ① pod 文件系统是临时的 → shca 用户态每个作业都要装一次（离线 dpkg -x，
#     包在共享存储免网络，幂等，~10s；官方 mlxtoshca 脚本要 apt 联网，pod 内不可用）
#  ② RCCL 认 shca 走的是外置插件 librccl-net-shca + 平台拓扑文件，不是内置 NET/IB
# ③ 镜像里版本号更高的 Mellanox 库(*47*)会遮住 shca 版(*44*)：必须删高版本实体、
#    把 SONAME 软链重指 shca 版，否则 ldconfig 不生效
# 资产缺失/安装失败一律自动回退 TCP（不阻塞已验证链路）；RDMA_DISABLE=1 可强制关闭。
enable_rdma() {
    if [[ "${RDMA_DISABLE:-0}" == "1" ]]; then
        log "RDMA: RDMA_DISABLE=1 → 保持 TCP(NET/Socket)"
        return
    fi
    local shca_home="${SHCA_HOME:-/root/private_data/second_hygon/rdma-shca}"
    local deb topo
    deb="$(ls "${shca_home}"/shca-tools_*.deb 2>/dev/null | head -1 || true)"
    topo="${shca_home}/topo_lib/built-in-508-topo-input-tj-default.xml"
    if [[ -z "$deb" || ! -f "$topo" || ! -d "${shca_home}/topo_lib/lib" ]]; then
        warn "RDMA: shca 资产不完整(${shca_home}) → 回退 TCP(NET/Socket)。大规模训练前建议补齐"
        return
    fi
    local libd=/usr/lib/x86_64-linux-gnu
    if [[ ! -e "${libd}/libshca-rdmav34.so" ]]; then
        log "安装 shca RDMA 用户态(离线 dpkg -x，约 10s): ${deb}"
        rm -rf /tmp/shca-install && mkdir -p /tmp/shca-install
        if ! dpkg -x "$deb" /tmp/shca-install 2>/dev/null; then
            warn "RDMA: dpkg -x 解包失败 → 回退 TCP"
            return
        fi
        cp -a "/tmp/shca-install${libd}/." "${libd}/"
        rm -f "${libd}/libibverbs.so.1.14.47.0" "${libd}/libmlx5.so.1.24.47.0" \
              "${libd}/librdmacm.so.1.3.47.0" "${libd}"/libibverbs.so.1.*47* 2>/dev/null || true
        ln -sf libibverbs.so.1.14.44.0 "${libd}/libibverbs.so.1"
        ln -sf librdmacm.so.1.3.44.0   "${libd}/librdmacm.so.1"
        ln -sf libmlx5.so.1.24.44.0    "${libd}/libmlx5.so.1"
        mkdir -p /etc/libibverbs.d /usr/etc/libibverbs.d /etc/cfgFile
        cp -a /tmp/shca-install/usr/etc/libibverbs.d/. /etc/libibverbs.d/ 2>/dev/null || true
        cp -a /tmp/shca-install/usr/etc/libibverbs.d/. /usr/etc/libibverbs.d/ 2>/dev/null || true
        cp -a /tmp/shca-install/etc/cfgFile/. /etc/cfgFile/ 2>/dev/null || true
        mkdir -p /usr/local/bin && cp -a /tmp/shca-install/usr/bin/. /usr/local/bin/ 2>/dev/null || true
        ldconfig
        if [[ ! -e "${libd}/libshca-rdmav34.so" ]]; then
            warn "RDMA: 安装后仍未找到 libshca-rdmav34.so → 回退 TCP"
            return
        fi
        log "shca 用户态安装完成(libshca-rdmav34.so 就位)"
    else
        log "RDMA: shca 用户态已就位(跳过安装)"
    fi
    # 管理员配方(vllm/sglang 同款)：外置插件 + 平台拓扑文件
    export NCCL_IB_DISABLE=0
    export NCCL_NET_PLUGIN=shca
    export NCCL_IB_HCA="${NCCL_IB_HCA:-shca_0:1,shca_1:1,shca_2:1,shca_3:1}"
    export NCCL_TOPO_FILE="$topo"
    export LD_LIBRARY_PATH="${shca_home}/topo_lib/lib:${LD_LIBRARY_PATH:-}"
    # 验证一眼：只有模型训练模块的作业挂 /dev/infiniband(notebook 不挂)。
    # 注意 ibv_devinfo 只是参考诊断——run-8 实测它没列出 shca，但 RCCL 实际
    # 全通道走了 IBext_v8；真正判据始终是训练日志的 NET/IBext_v8 / NET/Socket
    local hcas
    hcas="$(ibv_devinfo -l 2>/dev/null | grep '^shca_' | tr '\n' ' ' || true)"
    if [[ -n "$hcas" ]]; then
        log "RDMA: shca 设备可见:${hcas}。判据: NCCL_DEBUG=INFO 时日志应出现 NET/IBext_v8(出现 NET/Socket 即回退)"
    else
        log "RDMA: ibv_devinfo 未列出 shca(原始输出: $(ibv_devinfo -l 2>&1 | tr '\n' ' ' | head -c 160)) —— 仅供参考，最终以训练日志 NET/IBext_v8 为准"
    fi
}

start_training() {  # $1=node_rank  $2=master_addr
    log "启动训练: NODE_RANK=$1 NNODES=${NUM_NODES} MASTER_ADDR=$2:${MASTER_PORT}"
    log "成功判据: 本节点日志出现 iteration x/${TRAIN_ITERS} ... loss="
    # 平台日志是管道收集，python 子进程 stdout 默认块缓冲，不加这个训练日志
    # 要等进程退出才可见（上一轮 worker-0 静默的部分原因）。
    export PYTHONUNBUFFERED=1
    # K8s pod 用 FQDN 互相解析时可能命中多余地址，导致 RCCL 代理连接被对端
    # reset（socketProgress: Connection closed by remote peer / error code 6）。
    # 显式绑定容器主网卡（取默认路由出接口，失败回退 eth0）。
    # 注意 set -euo pipefail：命令替换里的管道失败会直接终止脚本（上一版踩中：
    # pod 镜像无 iproute2 → 127 → worker-0 在此行静默退出），必须 || true 兜底。
    local ifname=""
    if command -v ip >/dev/null 2>&1; then
        ifname="$(ip -o route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}' || true)"
    fi
    [[ -n "$ifname" ]] || ifname="eth0"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${ifname}}"
    # 镜像默认 NCCL_DEBUG=INFO：每节点刷几千行初始化日志（本轮 6000+ 行里
    # 大半是它，还拖慢迭代）。链路已验证通过，默认降到 WARN；需要排查时
    # 在平台环境变量里设 NCCL_DEBUG=INFO 即可恢复
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    # 生成脚本里默认 60 分钟；调短以便卡死时尽快拿到真实报错栈而非干等
    export DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-20}"
    log "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}  DIST_TIMEOUT_MIN=${DIST_TIMEOUT_MIN}"
    # 跨节点通道升级：TCP → shca RDMA（资产缺失/失败自动回退 TCP，见函数头注释）
    enable_rdma
    export NNODES="$NUM_NODES" NODE_RANK="$1" MASTER_ADDR="$2" MASTER_PORT GPUS_PER_NODE
    exec bash "$PLAN_SCRIPT"
}

# ── SCNet vcjob 主流程 ───────────────────────────────────────────────────────
vcjob_main() {
    local myip; myip="${WORKER0:-$(detect_local_ip)}"
    log "平台=SCNet vcjob  pod=${HOSTNAME:-$(hostname)}  role=worker-${ROLE}  本机IP=${myip}"
    case "$ROLE" in
        0)
            log "我是主节点(worker-0)"
            local_preflight
            ensure_submodules
            generate
            mkdir -p "$OUT_DIR"
            rm -f "${OUT_DIR}"/.vcjob_master_addr*   # 清理历史作业残留标记，防止无限累积
            printf '%s %s\n' "$JOB_TAG" "$myip" > "${ADDR_MARKER}.tmp" && mv "${ADDR_MARKER}.tmp" "$ADDR_MARKER"
            log "已发布主节点地址(作业${JOB_TAG}) → ${ADDR_MARKER}"
            start_training 0 "$myip"
            ;;
        1)
            log "我是从节点(worker-1)，等待主节点发布地址（最多 15 分钟）..."
            local waited=0 marker_tag=""
            while :; do
                if [[ -f "$ADDR_MARKER" ]]; then
                    marker_tag="$(awk '{print $1}' "$ADDR_MARKER" 2>/dev/null || true)"
                    # 只接受本作业的标记；旧作业遗留文件（无标记/标记不符）继续等
                    [[ "$marker_tag" == "$JOB_TAG" ]] && break
                fi
                sleep 5; waited=$((waited+5))
                [[ $waited -ge 900 ]] && die "等待本作业(${JOB_TAG})的地址标记超时：主节点生成失败或共享存储不可见，请查 worker-0 日志"
            done
            local master; master="$(awk '{print $2}' "$ADDR_MARKER")"
            [[ -n "$master" ]] || die "地址标记内容异常: $(cat "$ADDR_MARKER")"
            log "主节点地址: ${master}（作业标记 ${marker_tag} 吻合）"
            [[ -f "$PLAN_SCRIPT" ]] || die "共享存储上看不到 ${PLAN_SCRIPT}：挂载目录非共享或主节点未生成完成"
            local_preflight
            log "等待主节点 rendezvous 端口 ${master}:${MASTER_PORT} 开放..."
            waited=0
            until timeout 3 bash -c "echo > /dev/tcp/${master}/${MASTER_PORT}" 2>/dev/null; do
                sleep 5; waited=$((waited+5))
                [[ $waited -ge 900 ]] && { warn "端口等待超时，仍尝试启动（若失败请检查主节点日志）"; break; }
            done
            start_training 1 "$master"
            ;;
        *)
            die "未预期的 worker 序号: $ROLE（本脚本面向双机，worker-0/1）"
            ;;
    esac
}

# ── 裸机 SSH 模式（原有逻辑）─────────────────────────────────────────────────
ssh_ok() { ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "$1" "$2" >/dev/null 2>&1; }

detect_worker0() {
    if [[ -n "$WORKER0" ]]; then log "使用手动指定 worker0: ${WORKER0}"; return; fi
    local src
    src="$(ip route get "$WORKER1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}' || true)"
    [[ -z "$src" ]] && src="$(detect_local_ip)" && warn "路由侦测失败，回退 hostname -I: ${src}"
    WORKER0="$src"; log "自动侦测 worker0(本机): ${WORKER0}"
}

preflight() {
    local fail=0
    log "── 体检 1/6: SSH 免密（含本机自连）"
    for w in "$WORKER0" "$WORKER1"; do
        ssh_ok "$w" hostname && log "  SSH $w ✓" || { warn "  SSH $w ✗（先 ssh-copy-id $w）"; fail=1; }
    done
    [[ $fail -eq 0 ]] || die "SSH 不通，先解决免密"
    log "── 体检 2/6: 远端非交互 python3/torch"
    for w in "$WORKER0" "$WORKER1"; do
        local out
        if out="$(ssh -o BatchMode=yes "$w" "which python3 && python3 -c 'import torch; print(torch.__version__, torch.cuda.device_count())'" 2>&1)"; then
            log "  $w ✓ $(echo "$out" | tail -1)"
        else
            warn "  $w ✗ 非交互 SSH 下 python3/torch 不可用: $(echo "$out" | tail -1)"; fail=1
        fi
    done
    log "── 体检 3/6: 两台仓库路径一致（${REPO_ROOT}）"
    for w in "$WORKER0" "$WORKER1"; do
        ssh_ok "$w" "test -f '${REPO_ROOT}/tools/launch_cluster.py'" && log "  $w ✓" || { warn "  $w ✗ 路径不一致"; fail=1; }
    done
    log "── 体检 4/6: Megatron 子模块"
    for w in "$WORKER0" "$WORKER1"; do
        ssh_ok "$w" "test -f '${REPO_ROOT}/Megatron-LM-das/3rdparty/Megatron-LM/megatron/training/arguments.py'" \
            && log "  $w ✓" || { warn "  $w ✗ 需 git submodule update --init --recursive"; fail=1; }
    done
    log "── 体检 5/6: 模型目录"
    local model_abs; model_abs="$(cd "${REPO_ROOT}/${MODEL_PATH}" 2>/dev/null && pwd)" || die "本机缺模型目录"
    ssh_ok "$WORKER1" "test -d '${model_abs}'" && log "  worker1 ✓" || { warn "  worker1 ✗ 缺模型 → rsync -avP ${model_abs}/ ${WORKER1}:${model_abs}/"; fail=1; }
    log "── 体检 6/6: 端口 ${MASTER_PORT}"
    python3 - "$MASTER_PORT" <<'PYEOF' &
import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", int(sys.argv[1]))); s.listen(1); time.sleep(10)
PYEOF
    local listener=$!; sleep 1
    ssh -o BatchMode=yes "$WORKER1" "timeout 3 bash -c 'echo > /dev/tcp/${WORKER0}/${MASTER_PORT}'" 2>/dev/null \
        && log "  worker1 → ${WORKER0}:${MASTER_PORT} ✓" || { warn "  端口不通（防火墙？换 --master-port）"; fail=1; }
    kill "$listener" 2>/dev/null || true
    [[ $fail -eq 0 ]] || die "体检未通过（可 --skip-checks 强制）"
    log "── 体检全部通过"
}

fill_hostfile() {
    local hostfile="${OUT_DIR}/hostfile"
    sed -i.bak "s|^localhost slots=8  # node0|${WORKER0} slots=8|; s|^localhost slots=8  # node1|${WORKER1} slots=8|" "$hostfile"
    grep -q "^localhost" "$hostfile" && die "hostfile 仍有未替换的 localhost"
    log "hostfile 已写入 ${WORKER0} / ${WORKER1}"
}

smoke() {
    log "单节点冒烟（NNODES=1 规避 smoke 脚本默认值问题）"
    NNODES=1 NODE_RANK=0 GPUS_PER_NODE="$GPUS_PER_NODE" bash "${OUT_DIR}/launch_megatron_smoke.sh"
}

ssh_run() {
    mkdir -p "${REPO_ROOT}/logs/dual16_llama8b"
    log "双机拉起（MASTER_ADDR=${WORKER0}:${MASTER_PORT}）"
    python3 tools/launch_cluster.py \
        --hostfile "${OUT_DIR}/hostfile" \
        --script "$PLAN_SCRIPT" \
        --gpus-per-node "$GPUS_PER_NODE" --master-port "$MASTER_PORT" \
        --log-dir "${REPO_ROOT}/logs/dual16_llama8b"
}

stop() {
    local targets=(); [[ -n "${WORKER0:-}" ]] && targets+=("$WORKER0"); [[ -n "${WORKER1:-}" ]] && targets+=("$WORKER1")
    [[ ${#targets[@]} -eq 0 ]] && die "stop 需要 worker 地址或 vcjob 环境下本机执行"
    for w in "${targets[@]}"; do
        log "清理 $w ..."
        ssh -o BatchMode=yes "$w" "pkill -f pretrain_gpt; pkill -f torch.distributed.run; pkill -f run_megatron_hygon" 2>/dev/null || true
    done
}

ssh_main() {
    detect_worker0
    log "worker0=${WORKER0}  worker1=${WORKER1}  port=${MASTER_PORT}"
    case "$MODE" in
        check)    preflight ;;
        generate) [[ $SKIP_CHECKS -eq 1 ]] || preflight; generate; fill_hostfile ;;
        smoke)    smoke ;;
        run)      [[ $SKIP_CHECKS -eq 1 ]] || preflight; generate; fill_hostfile; ssh_run ;;
        stop)     stop ;;
        *)        die "未知 --mode: $MODE" ;;
    esac
}

# ── 入口：vcjob 优先，否则裸机 SSH 模式 ─────────────────────────────────────
main() {
    detect_repo_root
    log "仓库根: ${REPO_ROOT}"
    if is_vcjob && [[ "$MODE" == "auto" || "$MODE" == "run" ]]; then
        vcjob_main
    elif [[ "$MODE" == "auto" ]]; then
        [[ -n "$WORKER1" ]] || { usage; die "裸机模式必须提供 worker1 地址；SCNet vcjob 环境应自动识别（当前 hostname=$(hostname)）"; }
        ssh_main
    else
        [[ -n "$WORKER1" || "$MODE" == "stop" ]] || { usage; die "必须提供 worker1 地址（或 --mode stop）"; }
        ssh_main
    fi
}
main
