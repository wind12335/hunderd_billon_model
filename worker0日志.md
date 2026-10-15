日志

[dual-test] 脚本已启动 hostname=vcjob-2107528843008176130-abc12335-66689-worker-0 bash=5.1.16(1)-release
[1;32m[dual-test][0m 仓库根: /root/private_data/thousand_model
[1;32m[dual-test][0m 平台=SCNet vcjob  pod=vcjob-2107528843008176130-abc12335-66689-worker-0  role=worker-0  本机IP=172.31.50.213
[1;32m[dual-test][0m 我是主节点(worker-0)
[1;32m[dual-test][0m ── 本机 python3/torch
[1;32m[dual-test][0m   ✓ 2.7.1 cards: 8
[1;32m[dual-test][0m ── 本机模型目录 /root/private_data/thousand_model/model/Llama-3.1-8B
[1;32m[dual-test][0m   ✓ 在位
[1;32m[dual-test][0m ── Megatron 子模块 ✓ 已初始化
[1;32m[dual-test][0m 生成多机脚本（预期 world_size=16）...
============================================================
多机自动并行: 2 节点 × 8 卡 = 16 卡
模型: /root/private_data/thousand_model/model/Llama-3.1-8B
backend=MEGATRON run_mode=LOWER
搜索上限: world_size<=128, pp<=1, ep<=1
本机 GPU 数: 8（规划模式不要求匹配）
============================================================
------------------------------------------------------------
搜索结果: TP=4, DP=4, PP=1, EP=1, world_size=16, mbs=1, gbs=4, seq_parallel=True
预估单卡显存: 30.64 GB / 上限 64 GB
策略: BLOCK_LEVEL_FAST / BALANCED，耗时 0.0023s
------------------------------------------------------------
下放产物:
  smoke 脚本   : /root/private_data/thousand_model/generated/dual16_llama8b/launch_megatron_smoke.sh
  规划脚本     : /root/private_data/thousand_model/generated/dual16_llama8b/launch_megatron_plan.sh
  hostfile     : /root/private_data/thousand_model/generated/dual16_llama8b/hostfile
  一键拉起脚本 : /root/private_data/thousand_model/generated/dual16_llama8b/launch_cluster.sh
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
多机执行步骤:
  1. 编辑 hostfile，把 localhost 替换成真实节点名（第一行是主节点）。
  2. 确认所有节点在相同路径部署了本仓库且 SSH 免密。
  3. 在主节点执行: python3 /root/private_data/thousand_model/tools/launch_cluster.py --hostfile /root/private_data/thousand_model/generated/dual16_llama8b/hostfile --script /root/private_data/thousand_model/generated/dual16_llama8b/launch_megatron_plan.sh --gpus-per-node 8
  note: 运行时不自建 distributed runtime，而是把计划真实下放到现有 Megatron。
  note: 默认 launch_program 执行 smoke_test 脚本；规划脚本单独保留，等待后续真双卡环境。
  note: 规划脚本默认 WORLD_SIZE=16，smoke_test 默认 WORLD_SIZE=1。
  note: 当前 smoke test 默认回退到单卡串行配置，只验证数据链路和 Megatron 启动链路。
  note: manual/declarative 两个 py 文件已生成，可直接比较代码量。
  note: 已生成多机 hostfile 模板与 launch_cluster.sh 一键拉起脚本：请先把 hostfile 中的 localhost 替换成真实节点名，再在主节点执行 bash launch_cluster.sh。
[1;32m[dual-test][0m world_size=16 校验通过
[1;32m[dual-test][0m 已发布主节点地址(作业vcjob-2107528843008176130-abc12335-66689) → /root/private_data/thousand_model/generated/dual16_llama8b/.vcjob_master_addr.vcjob-2107528843008176130-abc12335-66689
[1;32m[dual-test][0m 启动训练: NODE_RANK=0 NNODES=2 MASTER_ADDR=172.31.50.213:23456
[1;32m[dual-test][0m 成功判据: 本节点日志出现 iteration x/20 ... loss=
[1;32m[dual-test][0m NCCL_SOCKET_IFNAME=eth  DIST_TIMEOUT_MIN=20
[megatron-hygon] MEGATRON_PROJECT_ROOT=/root/private_data/thousand_model
[megatron-hygon] MEGATRON_UPSTREAM_ROOT=/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM
[megatron-hygon] PYTHONPATH=/root/private_data/thousand_model/tools:/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM:/root/private_data/thousand_model
[megatron-hygon] 如需自检，请执行: python3 tools/check_megatron_hygon_env.py
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/hyper_comm_grid.py:26: UserWarning: absl.logging is not installed. Using logging.getLogger(__name__) instead. Please install absl.logging with `pip install absl-py` to use absl.logging.
  warnings.warn(
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
[megatron-hygon] 补丁层: hcu_megatron 源码（与上游 0.18.2 配套）: /root/private_data/thousand_model/Megatron-LM-das
using world size: 16, data-parallel size: 4, context-parallel size: 1, hierarchical context-parallel sizes: None, tensor-model-parallel size: 4, pipeline-model-parallel size: 1
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/training/utils/common_utils.py:414: UserWarning: Overriding default arguments for tokenizer_type:GPT2BPETokenizer with tokenizer_type:NullTokenizer
  warnings.warn(message)
Number of virtual stages per pipeline stage: None
accumulate and all-reduce gradients in fp32 for bfloat16 data type.
using torch.bfloat16 for parameters ...
------------------------ arguments ------------------------
  _explicit_args .................................. {'max_position_embeddings', 'tensorboard_dir', 'dist_url', 'adam_beta2', 'sequence_parallel', 'pipeline_model_parallel_size', 'eval_iters', 'min_lr', 'mock_data', 'local_rank', 'bf16', 'swiglu', 'num_query_groups', 'attention_dropout', 'untie_embeddings_and_output_weights', 'use_distributed_optimizer', 'lr_decay_style', 'micro_batch_size', 'lr', 'save', 'adam_beta1', 'seq_length', 'hidden_dropout', 'hidden_size', 'train_iters', 'tensor_model_parallel_size', 'distributed_timeout_minutes', 'tokenizer_type', 'add_bias_linear', 'eval_interval', 'clip_grad', 'log_interval', 'deprecated_use_mcore_models', 'world_size', 'rotary_base', 'weight_decay', 'vocab_size', 'recompute_method', 'normalization', 'ffn_hidden_size', 'rank', 'position_embedding_type', 'recompute_granularity', 'recompute_num_layers', 'global_batch_size', 'init_method_std', 'num_layers', 'save_interval', 'num_attention_heads'}
  _is_global_batch_size_explicitly_specified ...... True
  account_for_embedding_in_pipeline_split ......... False
  account_for_loss_in_pipeline_split .............. False
  accumulate_allreduce_grads_in_fp32 .............. True
  activation_func_clamp_value ..................... None
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.95
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. False
  add_position_embedding .......................... True
  add_qkv_bias .................................... False
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  align_grad_reduce ............................... True
  align_param_gather .............................. False
  allow_ambiguous_pad_tokens ...................... False
  app_tag_run_name ................................ None
  app_tag_run_version ............................. 0.0.0
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  apply_rope_fusion ............................... True
  apply_wd_to_qk_layernorm ........................ False
  async_ckpt_cpu_priority ......................... 10
  async_ckpt_io_priority .......................... 3
  async_ckpt_use_cpu_shm .......................... False
  async_save ...................................... False
  async_strategy .................................. mcore
  attention_backend ............................... AttnBackend.auto
  attention_dropout ............................... 0.0
  attention_output_gate ........................... False
  attention_softmax_in_fp32 ....................... False
  auto_detect_ckpt_format ......................... False
  barrier_with_L1_time ............................ True
  batch_invariant_mode ............................ False
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ True
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ False
  bias_swiglu_fusion .............................. True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  bridge_hf_model ................................. None
  bridge_language_model_only ...................... False
  cache_mla_latents ............................... False
  calc_ft_timeouts ................................ False
  calculate_per_token_loss ........................ False
  chat_template ................................... None
  check_for_large_grads ........................... False
  check_for_nan_in_loss_and_grad .................. True
  check_for_spiky_loss ............................ False
  check_weight_hash_across_dp_replicas_interval ... None
  ckpt_assume_constant_structure .................. False
  ckpt_convert_format ............................. None
  ckpt_convert_save ............................... None
  ckpt_convert_update_legacy_dist_opt_format ...... False
  ckpt_format ..................................... torch_dist
  ckpt_fully_parallel_load ........................ False
  ckpt_fully_parallel_load_exchange_algo .......... broadcast
  ckpt_fully_parallel_load_process_group .......... dp
  ckpt_fully_parallel_save ........................ True
  ckpt_fully_parallel_save_deprecated ............. False
  ckpt_fully_parallel_save_process_group .......... dp
  ckpt_load_validate_sharding_integrity ........... True
  ckpt_step ....................................... None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  clone_scatter_output_in_embedding ............... True
  collect_log_path ................................ ./logs
  comm_time_log_iter .............................. None
  config_logger_dir ............................... 
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  context_parallel_size ........................... 1
  cp_comm_type .................................... ['p2p']
  cpu_offloading_num_layers ....................... 0
  cpu_offloading_retain_pinned_cpu_buffers ........ False
  create_all_gather_group ......................... False
  create_attention_mask_in_dataloader ............. True
  cross_entropy_fusion_impl ....................... native
  cross_entropy_loss_fusion ....................... False
  cuda_graph_impl ................................. none
  cuda_graph_modules .............................. []
  cuda_graph_warmup_steps ......................... 3
  data_args_path .................................. None
  data_cache_path ................................. None
  data_parallel_random_init ....................... False
  data_parallel_sharding_strategy ................. optim_grads_params
  data_parallel_size .............................. 4
  data_path ....................................... None
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_defer_npy_index_mmap ................. False
  dataloader_fast_cache_load ...................... False
  dataloader_type ................................. single
  ddp_average_in_collective ....................... False
  ddp_bucket_size ................................. None
  ddp_num_buckets ................................. None
  ddp_pad_buckets_for_high_nccl_busbw ............. False
  ddp_param_name_patterns_for_fp32_local_accumulation  []
  ddp_reduce_scatter_with_fp32_accumulation ....... False
  decode_only_cuda_graphs ......................... False
  decoder_first_pipeline_num_layers ............... None
  decoder_last_pipeline_num_layers ................ None
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  decoupled_lr .................................... None
  decoupled_min_lr ................................ None
  decrease_batch_size_if_needed ................... False
  defer_embedding_wgrad_compute ................... False
  delay_1f1b_cooldown_wgrad_compute ............... False
  delay_wgrad_compute ............................. False
  deprecated_use_mcore_models ..................... True
  deterministic_mode .............................. False
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  disable_backward_fusion ......................... False
  disable_bf16_reduced_precision_matmul ........... False
  disable_jit_fuser ............................... False
  disable_msc_deprecated .......................... False
  disable_straggler_on_startup .................... False
  disable_symmetric_registration .................. False
  dist_ckpt_format_deprecated ..................... None
  dist_ckpt_optim_fully_reshardable ............... False
  dist_ckpt_save_pre_mcore_014 .................... False
  dist_ckpt_strictness ............................ assume_ok_unexpected
  dist_ckpt_workers ............................... 1
  dist_url ........................................ tcp://172.31.50.213:23456
  distrib_optim_fully_reshardable_mem_efficient ... False
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 20
  distributed_timeout_seconds_after_init .......... None
  dsa_indexer_head_dim ............................ None
  dsa_indexer_loss_coeff .......................... None
  dsa_indexer_n_heads ............................. None
  dsa_indexer_topk ................................ None
  dsa_indexer_use_sparse_loss ..................... False
  dump_param_to_param_group_map ................... None
  embedding_init_method_std ....................... None
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  enable_a2a_overlap_only_in_1f1b_phase ........... False
  enable_bw_flux_gemmrs_op ........................ True
  enable_chunked_prefill .......................... False
  enable_cuda_graph ............................... False
  enable_dynamic_grad_comp ........................ False
  enable_experimental ............................. False
  enable_ft_package ............................... False
  enable_full_sharding_in_hsdp .................... False
  enable_gpt_oss .................................. False
  enable_hyper_connections ........................ False
  enable_msc ...................................... False
  enable_one_logger ............................... True
  enable_pre_init_ep_overlap_streams .............. False
  enable_vocab_parallel ........................... False
  encoder_num_layers .............................. 32
  encoder_seq_length .............................. 4096
  end_weight_decay ................................ 0.1
  eod_mask_loss ................................... False
  ep_overlap_early_attn_memory_release ............ False
  ep_overlap_early_recompute ...................... False
  error_injection_rate ............................ 0
  error_injection_type ............................ transient_error
  eval_global_batch_size .......................... 4
  eval_interval ................................... 1
  eval_iters ...................................... 0
  eval_micro_batch_size ........................... 1
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal ..................................... 15
  exit_signal_handler ............................. False
  exit_signal_handler_for_dataloader .............. False
  exit_signal_handler_for_training ................ False
  exp_avg_dtype ................................... torch.float32
  exp_avg_sq_dtype ................................ torch.float32
  experimental_attention_variant .................. None
  expert_model_parallel_size ...................... 1
  expert_tensor_parallel_size ..................... 4
  export_default_te_spec .......................... False
  export_force_local_attention .................... False
  export_kd_cfg ................................... None
  export_kd_teacher_ckpt_format ................... None
  export_kd_teacher_load .......................... None
  export_kd_teacher_model_config .................. None
  export_kv_cache_quant ........................... none
  export_legacy_megatron .......................... False
  export_model_type ............................... GPTModel
  export_moe_apply_probs_on_input ................. False
  export_offline_model ............................ False
  export_qk_l2_norm ............................... False
  export_quant_cfg ................................ None
  export_real_quant_cfg ........................... None
  export_te_mcore_model ........................... False
  external_cuda_graph ............................. False
  extra_vocab_size ................................ 0
  fake_process_group .............................. False
  fault_injector_delay_start_iteration ............ None
  fault_injector_fault_delay ...................... None
  fault_injector_fault_probabilities .............. None
  fault_injector_fault_types ...................... None
  fault_injector_mtti_seconds ..................... None
  fault_injector_num_ranks ........................ None
  fault_injector_offset_seconds ................... None
  fault_injector_ranks ............................ None
  fault_injector_seed ............................. None
  ffn_hidden_size ................................. 14336
  fim_data ........................................ False
  fim_eod_token ................................... <|endoftext|>
  fim_fragment_rate ............................... None
  fim_middle_token ................................ <fim_middle>
  fim_no_prefix ................................... None
  fim_pad_token ................................... <fim_pad>
  fim_prefix_token ................................ <fim_prefix>
  fim_rate ........................................ 0.5
  fim_split_sample ................................ None
  fim_spm_rate .................................... 0.5
  fim_suffix_token ................................ <fim_suffix>
  fine_grained_activation_offloading .............. False
  fine_grained_offloading_max_inflight_offloads ... None
  finetune ........................................ False
  finetune_data_split ............................. train
  finetune_hf_dataset ............................. None
  first_last_layers_bf16 .......................... False
  flash_decode .................................... False
  flight_recorder_dump_on_timeout ................. True
  flight_recorder_dump_path ....................... None
  flight_recorder_extra_dump_on_exec .............. True
  flight_recorder_include_only_active ............. True
  flight_recorder_include_stack_trace ............. False
  flight_recorder_trace_buffer_size ............... 2000
  flux_transpose_weight ........................... False
  fp16 ............................................ False
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp4 ............................................. None
  fp4_param_gather ................................ False
  fp4_quantizer_factory ........................... None
  fp4_recipe ...................................... nvfp4
  fp8 ............................................. None
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_output_proj ................................. False
  fp8_param_gather ................................ False
  fp8_quantizer_factory ........................... None
  fp8_recipe ...................................... delayed
  fp8_wgrad ....................................... True
  fsdp_double_buffer .............................. False
  fsdp_manual_registration ........................ False
  ft_num_warmup_iters ............................. 5
  full_validation ................................. False
  fused_residual_rmsnorm .......................... False
  global_batch_size ............................... 4
  glu_linear_offset ............................... 0.0
  gpu_sniff_test_interval ......................... None
  grad_comp ....................................... False
  grad_comp_warm_up ............................... 0.1
  grad_reduce_in_bf16 ............................. False
  gradient_accumulation_fusion .................... True
  gradient_reduce_div_fusion ...................... True
  gradient_sample_ratio ........................... 1.0
  group_query_attention ........................... False
  grpo_clamp_eps_lower ............................ 0.01
  grpo_clamp_eps_upper ............................ 0.01
  grpo_entropy_term_weight ........................ 0.0
  grpo_filter_groups_with_same_reward ............. False
  grpo_group_size ................................. 2
  grpo_iterations ................................. 2
  grpo_kl_beta .................................... 0.001
  grpo_prompts_per_step ........................... 32
  head_lr_mult .................................... 1.0
  heterogeneous_layers_config_encoded_json ........ None
  heterogeneous_layers_config_path ................ None
  hidden_dropout .................................. 0.0
  hidden_size ..................................... 4096
  hierarchical_context_parallel_sizes ............. None
  high_priority_a2a_comm_stream ................... False
  high_priority_stream_groups ..................... []
  hybrid_context_parallel ......................... False
  hybrid_layer_pattern ............................ None
  hybrid_override_pattern ......................... None
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  image_tokens_per_sample ......................... None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... -1
  inference_coordinator_port ...................... None
  inference_cuda_graph_all_prefills ............... False
  inference_cuda_graph_scope ...................... InferenceCudaGraphScope.none
  inference_disable_ep_consensus .................. False
  inference_disable_triton_nvls_kernels ........... False
  inference_dynamic_batching ...................... False
  inference_dynamic_batching_block_size ........... 256
  inference_dynamic_batching_buffer_size_gb ....... 40.0
  inference_dynamic_batching_cuda_graph_mixed_prefill_count  16
  inference_dynamic_batching_enable_prefix_caching  False
  inference_dynamic_batching_mamba_memory_ratio ... None
  inference_dynamic_batching_max_requests ......... None
  inference_dynamic_batching_max_tokens ........... None
  inference_dynamic_batching_num_cuda_graphs ...... 16
  inference_dynamic_batching_paused_buffer_size_gb  None
  inference_dynamic_batching_prefix_caching_coordinator_policy  first_prefix_block
  inference_dynamic_batching_prefix_caching_eviction_policy  ref_zero
  inference_dynamic_batching_prefix_caching_mamba_gb  None
  inference_dynamic_batching_prefix_caching_routing_alpha  0.5
  inference_dynamic_batching_sampling_backend ..... torch
  inference_dynamic_batching_track_generated_token_events  False
  inference_dynamic_batching_track_paused_request_events  False
  inference_dynamic_batching_unified_memory_level . 0
  inference_fuse_tp_communication ................. False
  inference_grouped_gemm_backend .................. vllm
  inference_logging_step_interval ................. 0
  inference_max_requests .......................... 8
  inference_max_seq_length ........................ 2560
  inference_moe_disable_fused_quant_kernels ....... False
  inference_moe_token_dispatcher_type ............. nvls
  inference_rng_tracker ........................... False
  inference_text_gen_server_logging ............... False
  inference_use_synchronous_zmq_collectives ....... False
  inference_wandb_logging ......................... False
  init_method_std ................................. 0.02
  init_method_xavier_uniform ...................... False
  init_model_with_meta_device ..................... False
  initial_loss_scale .............................. 4294967296
  inprocess_active_world_size ..................... 16
  inprocess_barrier_timeout ....................... 120
  inprocess_completion_timeout .................... 120
  inprocess_empty_cuda_cache ...................... False
  inprocess_granularity ........................... node
  inprocess_hard_timeout .......................... 90
  inprocess_heartbeat_interval .................... 30
  inprocess_heartbeat_timeout ..................... 60
  inprocess_last_call_wait ........................ 1
  inprocess_max_iterations ........................ None
  inprocess_monitor_process_interval .............. 1.0
  inprocess_monitor_thread_interval ............... 1.0
  inprocess_progress_watchdog_interval ............ 1.0
  inprocess_restart ............................... False
  inprocess_soft_timeout .......................... 60
  inprocess_termination_grace_time ................ 1
  integrate_recompute_to_ep_comm_overlap .......... False
  is_hybrid_model ................................. False
  iter_per_epoch .................................. 1250
  iteration_sample_ratio .......................... 0.01
  iterations_to_skip .............................. []
  keep_fp8_transpose_cache ........................ False
  kitchen_attention_backend ....................... sdpa
  kv_channels ..................................... 128
  kv_lora_rank .................................... 32
  langrl_env_config ............................... None
  layernorm_epsilon ............................... 1e-05
  layernorm_zero_centered_gamma ................... False
  lazy_mpu_init ................................... False
  linear_attention_freq ........................... None
  linear_conv_kernel_dim .......................... 4
  linear_key_head_dim ............................. 128
  linear_num_key_heads ............................ 16
  linear_num_value_heads .......................... 32
  linear_value_head_dim ........................... 128
  lion_beta1 ...................................... 0.95
  lion_beta2 ...................................... 0.98
  load ............................................ None
  load_main_params_from_ckpt ...................... False
  load_weights .................................... False
  local_rank ...................................... 0
  log_device_memory_used .......................... False
  log_energy ...................................... False
  log_interval .................................... 1
  log_loss_scale_to_tensorboard ................... True
  log_max_attention_logit ......................... False
  log_memory_interval ............................. None
  log_memory_to_tensorboard ....................... False
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_progress .................................... False
  log_straggler ................................... False
  log_throughput .................................. False
  log_timers_to_tensorboard ....................... False
  log_validation_ppl_to_tensorboard ............... False
  log_world_size_to_tensorboard ................... False
  logging_level ................................... None
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 0.0001
  lr_decay_iters .................................. None
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. None
  lr_warmup_init .................................. 0.0
  lr_warmup_iters ................................. 0
  lr_warmup_samples ............................... 0
  lr_wsd_decay_iters .............................. None
  lr_wsd_decay_samples ............................ None
  lr_wsd_decay_style .............................. exponential
  main_grads_dtype ................................ torch.float32
  main_params_dtype ............................... torch.float32
  make_vocab_size_divisible_by .................... 128
  mamba_head_dim .................................. 64
  mamba_inference_conv_states_dtype ............... torch.bfloat16
  mamba_inference_ssm_states_dtype ................ torch.bfloat16
  mamba_num_groups ................................ 8
  mamba_num_heads ................................. None
  mamba_state_dim ................................. 128
  manual_gc ....................................... False
  manual_gc_eval .................................. True
  manual_gc_interval .............................. 0
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 131072
  max_seqlen_per_dp_cp_rank ....................... None
  max_tokens_to_oom ............................... 12000
  mega_moe ........................................ False
  megatron_fsdp_enable_fine_grained_param_gather .. False
  megatron_fsdp_grad_comm_dtype ................... None
  megatron_fsdp_main_grads_dtype .................. None
  megatron_fsdp_main_params_dtype ................. torch.float32
  memory_snapshot_path ............................ snapshot.pickle
  merge_file ...................................... None
  metadata_path ................................... None
  mhc_expand_emb .................................. False
  mhc_fix_muons ................................... False
  mhc_fuse_aggregate_compute ...................... False
  mhc_fuse_h_post_compute ......................... False
  mhc_hres_vwnstyle ............................... False
  mhc_init_gating_factor .......................... 0.01
  mhc_init_hpre_use_module_layer .................. False
  mhc_lite ........................................ False
  mhc_log_amax_per_step ........................... 1
  mhc_recompute_layer_num ......................... 1
  mhc_sinkhorn_iterations ......................... 1
  mhc_tau ......................................... 0.05
  mhc_use_tilekernels ............................. False
  micro_batch_size ................................ 1
  microbatch_group_size_per_vp_stage .............. None
  mid_level_dataset_surplus ....................... 0.005
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-05
  min_offloaded_tensor_size ....................... 1048576
  mla_down_proj_fusion ............................ False
  mlp_chunks_for_prefill .......................... 1
  mlp_chunks_for_training ......................... 1
  mmap_bin_files .................................. True
  mock_data ....................................... True
  modelopt_enabled ................................ False
  moe_apply_probs_on_input ........................ False
  moe_aux_loss_coeff .............................. 0.0
  moe_deepep_num_sms .............................. 20
  moe_enable_deepep ............................... False
  moe_enable_routing_replay ....................... False
  moe_enable_ultraep .............................. False
  moe_expert_capacity_factor ...................... None
  moe_expert_rank_capacity_factor ................. None
  moe_ffn_hidden_size ............................. None
  moe_flex_dispatcher_backend ..................... deepep
  moe_grouped_gemm ................................ False
  moe_hybridep_num_blocks_permute ................. None
  moe_hybridep_num_blocks_unpermute ............... None
  moe_hybridep_num_sms ............................ None
  moe_hybridep_num_sms_preprocessing .............. 108
  moe_input_jitter_eps ............................ None
  moe_latent_size ................................. None
  moe_layer_freq .................................. 1
  moe_layer_recompute ............................. False
  moe_mlp_glu_interleave_size ..................... None
  moe_num_redundant_experts_per_rank .............. 0
  moe_pad_expert_input_to_capacity ................ False
  moe_pad_experts_for_cuda_graph_inference ........ False
  moe_paged_stash ................................. False
  moe_paged_stash_buffer_size_factor_cpu .......... 0.0
  moe_paged_stash_buffer_size_factor_cuda ......... 1.1
  moe_paged_stash_page_size ....................... 64
  moe_per_layer_logging ........................... False
  moe_permute_fusion .............................. False
  moe_permute_fusion_into_hybridep ................ False
  moe_router_bias_update_rate ..................... 0.001
  moe_router_dtype ................................ None
  moe_router_enable_expert_bias ................... False
  moe_router_force_biased ......................... None
  moe_router_force_load_balancing ................. False
  moe_router_fusion ............................... False
  moe_router_group_topk ........................... None
  moe_router_load_balancing_type .................. aux_loss
  moe_router_num_groups ........................... None
  moe_router_padding_for_fp8 ...................... False
  moe_router_padding_for_quantization ............. False
  moe_router_pre_softmax .......................... False
  moe_router_score_function ....................... softmax
  moe_router_topk ................................. 2
  moe_router_topk_scaling_factor .................. None
  moe_shared_expert_gate .......................... False
  moe_shared_expert_intermediate_size ............. None
  moe_shared_expert_overlap ....................... False
  moe_single_grouped_bias ......................... False
  moe_single_grouped_weight ....................... False
  moe_token_dispatcher_type ....................... allgather
  moe_token_drop_policy ........................... probs
  moe_ultraep_autotune ............................ False
  moe_ultraep_autotune_grad_reduce_max_sms ........ None
  moe_ultraep_autotune_start_iteration ............ 3
  moe_upcycling_granularity ....................... 1
  moe_use_upcycling ............................... False
  moe_z_loss_coeff ................................ None
  mrope_section ................................... None
  mscale .......................................... 1.0
  mscale_all_dim .................................. 0.0
  mtp_hybrid_override_pattern ..................... None
  mtp_loss_scaling_factor ......................... 0.1
  mtp_num_layers .................................. None
  mtp_standalone .................................. False
  mtp_use_repeated_layer .......................... False
  multi_latent_attention .......................... False
  multiple_validation_sets ........................ False
  muon_coefficient_type ........................... quintic
  muon_extra_scale_factor ......................... 1.0
  muon_fp32_matmul_prec ........................... medium
  muon_momentum ................................... 0.9
  muon_nesterov ................................... False
  muon_num_ns_steps ............................... 5
  muon_scalar_optimizer ........................... adam
  muon_scale_mode ................................. spectral
  muon_split_qkv .................................. True
  muon_tp_mode .................................... blockwise
  mup_attn_scale_power ............................ 1.0
  mup_base_head_dim ............................... None
  mup_base_hidden_size ............................ None
  mup_embedding_mult .............................. 1.0
  mup_output_mult ................................. 1.0
  mup_width_mult .................................. 1.0
  nccl_all_reduce_for_prefill ..................... False
  nccl_communicator_config_path ................... None
  nccl_ub ......................................... False
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_rope_freq .................................... None
  no_save_optim ................................... None
  no_save_rng ..................................... None
  no_weight_decay_cond_type ....................... None
  non_persistent_ckpt_type ........................ None
  non_persistent_global_ckpt_dir .................. None
  non_persistent_local_ckpt_algo .................. fully_parallel
  non_persistent_local_ckpt_dir ................... None
  non_persistent_save_interval .................... None
  normalization ................................... RMSNorm
  null_tokenizer_eod_id ........................... None
  null_tokenizer_pad_id ........................... -1
  num_attention_heads ............................. 32
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_dataset_builder_threads ..................... 1
  num_distributed_optimizer_instances ............. 1
  num_experts ..................................... None
  num_layers ...................................... 32
  num_layers_at_end_in_bf16 ....................... 1
  num_layers_at_start_in_bf16 ..................... 1
  num_layers_per_virtual_pipeline_stage ........... None
  num_layers_to_build ............................. None
  num_query_groups ................................ 8
  num_residual_streams ............................ 4
  num_speculative_tokens .......................... 0
  num_virtual_stages_per_pipeline_rank ............ None
  num_workers ..................................... 2
  nvtx_ranges ..................................... False
  object_storage_cache_path ....................... None
  offload_modules ................................. []
  one_logger_async ................................ False
  one_logger_project .............................. megatron-lm
  one_logger_run_name ............................. None
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  optimizer_cpu_offload ........................... False
  optimizer_cuda_graph ............................ False
  optimizer_offload_fraction ...................... 1.0
  outer_dp_sharding_strategy ...................... no_shard
  output_bert_embeddings .......................... False
  overlap_cpu_optimizer_d2h_h2d ................... False
  overlap_dispatch_backward_with_experts_wgrad .... False
  overlap_ep_comm_with_split_attn ................. False
  overlap_grad_reduce ............................. False
  overlap_moe_expert_parallel_comm ................ False
  overlap_p2p_comm ................................ False
  overlap_p2p_comm_warmup_flush ................... False
  overlap_param_gather ............................ False
  overlap_param_gather_with_optimizer_step ........ False
  override_opt_param_scheduler .................... False
  pad_vocab_size .................................. True
  padded_vocab_size ............................... None
  parallel_linear_impl ............................ None
  params_dtype .................................... torch.bfloat16
  patch_dim ....................................... 16
  per_dataset_sequences_path ...................... None
  per_split_data_args_path ........................ None
  perform_initialization .......................... True
  perform_rl_step ................................. False
  phase_transition_iterations ..................... None
  pin_cpu_grads ................................... True
  pin_cpu_params .................................. True
  pipe_sp_splits .................................. 1
  pipe_sp_strategy ................................ average
  pipeline_model_parallel_comm_backend ............ None
  pipeline_model_parallel_layout .................. None
  pipeline_model_parallel_size .................... 1
  position_embedding_type ......................... rope
  pretrained_checkpoint ........................... None
  profile ......................................... False
  profile_dir ..................................... ./
  profile_ranks ................................... []
  profile_step_end ................................ 12
  profile_step_start .............................. 10
  pytorch_profiler_collect_callstack .............. False
  pytorch_profiler_collect_chakra ................. False
  pytorch_profiler_collect_shapes ................. False
  q_lora_rank ..................................... None
  qk_clip ......................................... False
  qk_clip_alpha ................................... 0.5
  qk_clip_threshold ............................... 100
  qk_head_dim ..................................... 128
  qk_l2_norm ...................................... False
  qk_layernorm .................................... False
  qk_pos_emb_head_dim ............................. 64
  quant_comm_bits ................................. 8
  quant_group_size ................................ None
  quant_scale_dtype ............................... bf16
  query_in_block_prob ............................. 0.1
  quick_geglu ..................................... False
  rampup_batch_size ............................... None
  rank ............................................ 0
  rank_adjust_window_size ......................... 1000
  recompute_activation_function ................... False
  recompute_activation_function_num_layers ........ None
  recompute_granularity ........................... full
  recompute_in_advance ............................ False
  recompute_in_bubble ............................. False
  recompute_layer_ids ............................. None
  recompute_method ................................ uniform
  recompute_modules ............................... None
  recompute_mtp_layer_ids ......................... None
  recompute_num_layers ............................ 1
  record_memory_history ........................... False
  record_shapes ................................... False
  reduce_recompute_for_last_chunk ................. False
  refit_method .................................... gloo
  relative_attention_max_distance ................. 128
  relative_attention_num_buckets .................. 32
  replication ..................................... False
  replication_factor .............................. 2
  replication_jump ................................ None
  reproduce ....................................... False
  rerun_mode ...................................... validate_results
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  result_rejected_tracker_filename ................ None
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  reuse_fp32_param ................................ False
  reuse_grad_buf_for_mxfp8_param_ag ............... False
  rl_default_temperature .......................... 1.0
  rl_default_top_k ................................ -1
  rl_default_top_p ................................ 0
  rl_generation_batch_size ........................ None
  rl_importance_sampling_truncation_coef .......... None
  rl_inference_expert_model_parallel_size ......... None
  rl_inference_expert_tensor_model_parallel_size .. None
  rl_inference_logprobs_is_correction ............. False
  rl_inference_model_unified_memory_level ......... 0
  rl_inference_parsers ............................ []
  rl_inference_pipeline_model_parallel_size ....... None
  rl_inference_tensor_model_parallel_size ......... None
  rl_kv_cache_management_mode ..................... persist
  rl_num_parallel_generation_batches .............. None
  rl_num_parallel_generations ..................... None
  rl_offload_inference_model_weights_when_idle .... False
  rl_offload_optimizer_during_inference ........... False
  rl_parallel_generation_tasks .................... None
  rl_partial_rollouts ............................. False
  rl_persist_cuda_graphs .......................... False
  rl_prompts_per_eval ............................. 32
  rl_sequence_packing_algo ........................ fifo
  rl_sequence_packing_max_sequences_per_bin ....... 50
  rl_skip_bos_token ............................... False
  rl_training_cuda_graphs ......................... False
  rl_use_sequence_packing ......................... False
  rl_verify_model_weights_swap .................... False
  rope_scaling_factor ............................. 8.0
  rope_type ....................................... None
  rotary_base ..................................... 500000
  rotary_interleaved .............................. False
  rotary_percent .................................. 1.0
  rotary_scaling_factor ........................... 1.0
  rotary_seq_len_interpolation_factor ............. None
  run_workload_inspector_server ................... False
  sample_rate ..................................... 1.0
  save ............................................ /root/private_data/thousand_model/generated/autoparallel/checkpoints
  save_activations_interval ....................... None
  save_dgrads_interval ............................ None
  save_flux_gather_input .......................... False
  save_hf_weights ................................. False
  save_hf_weights_distributed ..................... False
  save_interval ................................... 1000
  save_params_interval ............................ None
  save_retain_interval ............................ None
  save_tokens_per_expert_interval ................. None
  save_wgrads_interval ............................ None
  schedule_method ................................. vanilla
  schedule_timer_end .............................. 20
  schedule_timer_start ............................ 10
  seed ............................................ 1234
  seq_length ...................................... 4096
  sequence_parallel ............................... True
  sft ............................................. False
  sft_tokenizer_prompt_format ..................... nemotron-h-aligned
  sgd_momentum .................................... 0.9
  sharp_enabled_group ............................. None
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  skipped_train_samples ........................... 0
  softmax_type .................................... vanilla
  spec ............................................ None
  special_tokens .................................. None
  specify_layers .................................. None
  split ........................................... None
  squared_relu .................................... False
  start_eval_at_iter .............................. None
  start_weight_decay .............................. 0.1
  step_batch_size_schedule ........................ None
  straggler_ctrlr_port ............................ 65535
  straggler_minmax_count .......................... 1
  strict_fsdp_dtensor_load ........................ True
  suggested_communication_unit_size ............... None
  swap_attention .................................. False
  swap_modules .................................... self_attention
  swiglu .......................................... True
  swin_backbone_type .............................. tiny
  symmetric_ar_type ............................... None
  sync_free_moe ................................... False
  sync_free_moe_backend ........................... primus
  te_precision_config_file ........................ None
  te_rng_tracker .................................. False
  tensor_model_parallel_size ...................... 4
  tensorboard_dir ................................. /root/private_data/thousand_model/generated/autoparallel/tensorboard
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1000
  test_data_path .................................. None
  test_mode ....................................... False
  tiktoken_num_special_tokens ..................... 1000
  tiktoken_pattern ................................ None
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_hf_no_include_special_tokens .......... False
  tokenizer_hf_no_use_fast ........................ False
  tokenizer_model ................................. None
  tokenizer_sentencepiece_ignore_extra_whitespaces  True
  tokenizer_sentencepiece_legacy .................. False
  tokenizer_type .................................. NullTokenizer
  torch_fsdp2_reshard_after_forward ............... True
  tp_comm_bootstrap_backend ....................... nccl
  tp_comm_bulk_dgrad .............................. True
  tp_comm_bulk_wgrad .............................. True
  tp_comm_overlap ................................. False
  tp_comm_overlap_ag .............................. True
  tp_comm_overlap_cfg ............................. None
  tp_comm_overlap_rs .............................. True
  tp_comm_overlap_rs_dgrad ........................ False
  tp_comm_split_ag ................................ True
  tp_comm_split_rs ................................ True
  train_data_path ................................. None
  train_iters ..................................... 20
  train_samples ................................... None
  train_sync_interval ............................. None
  transformer_impl ................................ transformer_engine
  transformer_pipeline_model_parallel_size ........ 1
  trust_remote_code ............................... False
  turbo_deepep_num_cu ............................. 32
  turbo_deepep_use_comm_stream .................... False
  turbo_mega_moe_precision ........................ bf16
  turbo_sync_free_moe_stage ....................... None
  untie_embeddings_and_output_weights ............. True
  use_bridge ...................................... False
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_ckpt_memory_cache ........................... False
  use_cpu_initialization .......................... None
  use_dist_ckpt ................................... True
  use_dist_ckpt_deprecated ........................ False
  use_distributed_optimizer ....................... True
  use_flash_attn .................................. False
  use_fused_weighted_squared_relu ................. False
  use_gloo_process_groups ......................... True
  use_grouped_gemm_for_dense_mlp .................. False
  use_hip_profiler ................................ False
  use_kitchen_attention ........................... False
  use_layer_wise_distributed_optimizer ............ False
  use_layer_wise_param_layout ..................... True
  use_legacy_static_engine ........................ False
  use_mamba_mem_eff_path .......................... True
  use_megatron_fsdp ............................... False
  use_mhc_svd ..................................... False
  use_mp_args_from_checkpoint_args ................ False
  use_mup ......................................... False
  use_one_sent_docs ............................... False
  use_optimizer_feature ........................... False
  use_persistent_ckpt_worker ...................... False
  use_precision_aware_optimizer ................... False
  use_primus_deepep ............................... False
  use_primus_fused_act_with_probs ................. False
  use_primus_grouped_gemm ......................... False
  use_primus_moe_permute_fusion ................... False
  use_primus_topk_router .......................... False
  use_pytorch_profiler ............................ False
  use_quantize_comm ............................... False
  use_ring_exchange_p2p ........................... False
  use_rope_scaling ................................ False
  use_rotary_position_embeddings .................. False
  use_sharp ....................................... False
  use_te_activation_func .......................... False
  use_tokenizer_model_from_checkpoint_args ........ True
  use_torch_fsdp2 ................................. False
  use_torch_optimizer_for_cpu_offload ............. False
  use_tp_pp_dp_mapping ............................ False
  use_transformer_engine_op_fuser ................. False
  use_vwn ......................................... False
  v_head_dim ...................................... 128
  valid_data_path ................................. None
  validation_set_names ............................ None
  variable_seq_lengths ............................ False
  verify_integrity ................................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... None
  vocab_size ...................................... 32000
  wandb_entity .................................... None
  wandb_exp_name .................................. None
  wandb_project ................................... None
  wandb_save_dir .................................. None
  weight_decay .................................... 0.1
  weight_decay_incr_style ......................... constant
  wgrad_deferral_limit ............................ 0
  window_attn_skip_freq ........................... None
  window_size ..................................... None
  world_size ...................................... 16
  yaml_cfg ........................................ None
-------------------- end of arguments ---------------------
------------------------ env vars ------------------------
  _ ............................................... /usr/local/bin/python3
  AMDGPU_TARGETS .................................. gfx906;gfx926;gfx928;gfx936;gfx938
  AUTOPARALLEL_ROLE ............................... planning
  C_INCLUDE_PATH .................................. /opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/opencl/include:
  CHECKPOINT_DIR .................................. /root/private_data/thousand_model/generated/autoparallel/checkpoints
  CHOWN_HOME ...................................... yes
  CHOWN_HOME_OPTS ................................. -R
  CMAKE_PREFIX_PATH ............................... /opt/hyhal:
  CPLUS_INCLUDE_PATH .............................. /opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/dushmem/include:/opt/dtk/opencl/include:
  CUDA_DEVICE_MAX_CONNECTIONS ..................... 1
  CUDA_MODULE_LOADING ............................. LAZY
  DATA_CACHE_PATH ................................. /root/private_data/thousand_model/generated/dual16_llama8b/data_cache
  DEBIAN_FRONTEND ................................. noninteractive
  DIST_TIMEOUT_MIN ................................ 20
  DTK_HOME ........................................ /opt/dtk
  DTKROOT ......................................... /opt/dtk
  FLASH_MLA_ROOT_DIR .............................. /usr/local/lib/python3.11/site-packages/flash_mla/asm/
  FLUX_ROOT_DIR_GFX936 ............................ /usr/local/lib/python3.11/site-packages/flux/blas_gemm/gfx936/
  FLUX_ROOT_DIR_GFX938 ............................ /usr/local/lib/python3.11/site-packages/flux/blas_gemm/gfx938/
  ftp_proxy ....................................... http://preset:6e298f07@10.16.1.51:3128
  GPUS_PER_NODE ................................... 8
  GROUP_RANK ...................................... 0
  GROUP_WORLD_SIZE ................................ 2
  HF_ENDPOINT ..................................... https://hf-mirror.com
  HIP_PATH ........................................ /opt/dtk/hip
  HOME ............................................ /root
  HOSTNAME ........................................ vcjob-2107528843008176130-abc12335-66689-worker-0
  http_proxy ...................................... http://preset:6e298f07@10.16.1.51:3128
  https_proxy ..................................... http://preset:6e298f07@10.16.1.51:3128
  HYHAL_PATH ...................................... /opt/hyhal
  JUPYTER_ENABLE_LAB .............................. yes
  JUPYTERLAB_WORKSPACES_DIR ....................... /tmp/
  KUBERNETES_PORT ................................. tcp://172.30.0.1:443
  KUBERNETES_PORT_443_TCP ......................... tcp://172.30.0.1:443
  KUBERNETES_PORT_443_TCP_ADDR .................... 172.30.0.1
  KUBERNETES_PORT_443_TCP_PORT .................... 443
  KUBERNETES_PORT_443_TCP_PROTO ................... tcp
  KUBERNETES_SERVICE_HOST ......................... 172.30.0.1
  KUBERNETES_SERVICE_PORT ......................... 443
  KUBERNETES_SERVICE_PORT_HTTPS ................... 443
  LANG ............................................ en_US.UTF-8
  LANGUAGE ........................................ en_US.UTF-8
  LC_ALL .......................................... en_US.UTF-8
  LD_LIBRARY_PATH ................................. /opt/dtk/lib:/opt/dtk/lib64:/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:/usr/local/lib/:/usr/local/lib64/:
  LIBRARY_PATH .................................... /opt/hyhal/lib:/opt/dtk/opencl/lib:/opt/dtk/dushmem/lib:
  LOCAL_RANK ...................................... 0
  LOCAL_WORLD_SIZE ................................ 8
  MANPATH ......................................... /opt/mpi/share/man:
  MASTER_ADDR ..................................... 172.31.50.213
  MASTER_PORT ..................................... 23456
  PYTHONPATH ...................................... /root/private_data/thousand_model/tools:/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM:/root/private_data/thousand_model
  PYTHONUNBUFFERED ................................ 1
  RANK ............................................ 0
  ROCM_HOME ....................................... /opt/dtk
  ROCM_PATH ....................................... /opt/dtk
  ROLE_NAME ....................................... default
  ROLE_RANK ....................................... 0
  ROLE_WORLD_SIZE ................................. 16
  SCNET_SHARING_CENTER ............................ /public/SothisAI/sharingCenter
  SCNET_TEAM_SHARE ................................ /public/share/abc12335
  SCNET_USER_HOME ................................. /public/home/abc12335
  SHELL ........................................... /bin/bash
  SHLVL ........................................... 1
  TASK_PATH ....................................... /public/home/abc12335/.SothisAI/k8s_instance/ssh/vcjob-2107528843008176130-abc12335-66689-worker-0/2107528843008176131/
  TENSORBOARD_DIR ................................. /root/private_data/thousand_model/generated/autoparallel/tensorboard
  TORCH_COMPILE_DISABLE ........................... 1
  TORCH_NCCL_ASYNC_ERROR_HANDLING ................. 1
  TORCHDYNAMO_DISABLE ............................. 1
  TORCHELASTIC_ERROR_FILE ......................... /tmp/torchelastic_sxgmdol7/none_966tidts/attempt_0/0/error.json
  TORCHELASTIC_MAX_RESTARTS ....................... 0
  TORCHELASTIC_RESTART_COUNT ...................... 0
  TORCHELASTIC_RUN_ID ............................. none
  TORCHELASTIC_USE_AGENT_STORE .................... True
  TORCHINDUCTOR_CACHE_DIR ......................... /tmp/torchinductor_root
  TZ .............................................. CST-8
  UCX_HOME ........................................ /opt/ucx
  VC_WORKER_HOSTS ................................. vcjob-2107528843008176130-abc12335-66689-worker-0.vcjob-2107528843008176130-abc12335-66689,vcjob-2107528843008176130-abc12335-66689-worker-1.vcjob-2107528843008176130-abc12335-66689
  VC_WORKER_NUM ................................... 2
  worker0 ......................................... vcjob-2107528843008176130-abc12335-66689-worker-0
  worker1 ......................................... vcjob-2107528843008176130-abc12335-66689-worker-1
  WORLD_SIZE ...................................... 16
-------------------- end of env vars ---------------------
/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/training/argument_utils.py:441: UserWarning: 
        It is recommended to use a ModelConfig (e.g. megatron.training.models.HybridModelConfig) instead
        of a model builder/model provider function pointer.
        
  warnings.warn(msg)
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.637844   356 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 6] ProcessGroupNCCL initialization options: size: 16, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:38.637876   356 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 6] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.637967   352 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 2] ProcessGroupNCCL initialization options: size: 16, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.637969   354 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 4] ProcessGroupNCCL initialization options: size: 16, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:38.638002   352 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.638007   354 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 4] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.638700   356 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 5 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 5
I1007 01:51:38.638710   356 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 5 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.638775   354 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 1 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 1
I1007 01:51:38.638788   354 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 1 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.638945   352 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 5 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 5
I1007 01:51:38.638959   352 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 5 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.643613   351 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 1] ProcessGroupNCCL initialization options: size: 16, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:38.643667   351 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.645704   351 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 3 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 3
I1007 01:51:38.645717   351 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 3 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
[WARNING  | megatron.core.rerun_state_machine]: RerunStateMachine initialized in mode RerunMode.VALIDATE_RESULTS
> initializing torch distributed ...
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.672061   350 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 0] ProcessGroupNCCL initialization options: size: 16, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:38.672101   350 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.672871   350 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 1 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 1
I1007 01:51:38.672884   350 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 1 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.720182   353 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 3] ProcessGroupNCCL initialization options: size: 16, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:38.720216   353 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.721483   353 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 7 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 7
I1007 01:51:38.721498   353 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 7 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.900911   350 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 9 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 9
I1007 01:51:38.900929   350 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 9 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.901678   354 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 9 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 9
I1007 01:51:38.901706   354 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 9 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.903725   350 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 17 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 17
I1007 01:51:38.903739   350 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 17 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.904258   354 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 21 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 21
I1007 01:51:38.904274   354 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 21 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.904371   350 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 33 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 33
I1007 01:51:38.904383   350 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 33 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.904738   350 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 37 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 37
I1007 01:51:38.904750   350 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 37 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.904871   354 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 34 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 34
I1007 01:51:38.904884   354 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 34 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.905117   350 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 41 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 41
I1007 01:51:38.905129   350 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 41 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.905259   354 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 38 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 38
I1007 01:51:38.905272   354 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 38 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.905315   350 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 42 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 42
I1007 01:51:38.905325   350 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 42 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.905499   350 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 43 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 43
I1007 01:51:38.905510   350 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 43 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.905941   354 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 53 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 53
I1007 01:51:38.905953   354 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 53 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.906147   354 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 54 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 54
I1007 01:51:38.906159   354 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 54 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.906344   354 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 55 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 55
I1007 01:51:38.906355   354 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 55 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.906857   350 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 0] ProcessGroupNCCL initialization options: size: 16, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:38.906869   350 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.907114   350 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 0] ProcessGroupNCCL initialization options: size: 16, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:38.907125   350 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.907371   350 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 91 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 91
I1007 01:51:38.907382   350 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 91 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.907738   350 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 95 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 95
I1007 01:51:38.907750   350 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 95 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.908353   350 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 111 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 111
I1007 01:51:38.908365   350 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 111 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.908694   350 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 115 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 115
I1007 01:51:38.908704   350 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 115 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.909036   350 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 119 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 119
I1007 01:51:38.909049   350 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 119 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.909900   350 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 123 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 123
I1007 01:51:38.909930   350 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 123 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.914179   354 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 4] ProcessGroupNCCL initialization options: size: 16, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:38.914193   354 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 4] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.914544   354 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 4] ProcessGroupNCCL initialization options: size: 16, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:38.914556   354 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 4] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.914868   354 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 92 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 92
I1007 01:51:38.914880   354 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 92 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.915350   354 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 99 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 99
I1007 01:51:38.915361   354 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 99 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.915921   354 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 112 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 112
I1007 01:51:38.915933   354 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 112 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.916281   354 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 116 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 116
I1007 01:51:38.916291   354 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 116 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.916646   354 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 120 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 120
I1007 01:51:38.916658   354 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 120 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.916972   354 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 123 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 123
I1007 01:51:38.916985   354 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 123 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.918692   350 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 0] ProcessGroupNCCL initialization options: size: 16, global rank: 0, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:38.918705   350 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
tp_group: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
pp_group: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
dp_group: [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
ep_group: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
etp_group: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
edp_group: [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
cp_group: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
tp-cp_group: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
embd-pp_group: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
pos_embd-pp_group: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
tp-ep_group: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
tp-dp-cp_group: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
tp-pp_group: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
dp-cp_group: [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
> initialized tensor model parallel with size 4
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
I1007 01:51:38.919023   354 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 4] ProcessGroupNCCL initialization options: size: 16, global rank: 4, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:38.919039   354 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 4] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
> compiling dataset index builder ...
make: Entering directory '/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/datasets'
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:38.927374   357 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 7] ProcessGroupNCCL initialization options: size: 16, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:38.927415   357 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 7] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:38.928540   357 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 7 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 7
I1007 01:51:38.928553   357 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 7 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
make: Nothing to be done for 'default'.
make: Leaving directory '/root/private_data/thousand_model/Megatron-LM-das/3rdparty/Megatron-LM/megatron/core/datasets'
>>> done with dataset index builder. Compilation time: 0.051 seconds
/usr/local/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4631: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
  warnings.warn(  # warn only once
W1007 01:51:39.007556   354 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 4]  using GPU 4 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:39.014254   354 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 4] Using non-blocking mode: 0
W1007 01:51:39.024538   350 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:39.028419   350 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
I1007 01:51:39.029302   350 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.04583 ms
I1007 01:51:39.029345   350 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
I1007 01:51:39.029452   354 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 4] ProcessGroupNCCL broadcast unique ID through store took 15.1784 ms
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:51:39.029492   354 NCCLUtils.cpp:75] Rank 4: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Comm config Blocking set to 1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Comm config Blocking set to 1
I1007 01:51:39.052838   356 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 13 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 13
I1007 01:51:39.052855   356 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 13 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.052915   352 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 13 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 13
I1007 01:51:39.052933   352 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 13 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.055176   352 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 19 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 19
I1007 01:51:39.055207   352 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 19 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.055476   356 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 23 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 23
I1007 01:51:39.055487   356 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 23 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.055811   352 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 33 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 33
I1007 01:51:39.055826   352 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 33 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.056015   356 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 34 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 34
I1007 01:51:39.056026   356 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 34 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.056186   352 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 37 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 37
I1007 01:51:39.056198   352 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 37 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.056382   356 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 38 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 38
I1007 01:51:39.056392   356 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 38 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.056737   352 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 47 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 47
I1007 01:51:39.056749   352 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 47 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.056960   352 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 48 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 48
I1007 01:51:39.056973   352 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 48 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.057152   352 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 49 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 49
I1007 01:51:39.057163   352 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 49 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.057209   356 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 59 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 59
I1007 01:51:39.057219   356 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 59 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.057405   356 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 60 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 60
I1007 01:51:39.057415   356 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 60 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.057582   356 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 61 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 61
I1007 01:51:39.057592   356 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 61 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.058351   352 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 2] ProcessGroupNCCL initialization options: size: 16, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:39.058363   352 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.058504   356 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 6] ProcessGroupNCCL initialization options: size: 16, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:39.058516   356 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 6] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.058622   352 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 2] ProcessGroupNCCL initialization options: size: 16, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:39.058634   352 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.058784   356 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 6] ProcessGroupNCCL initialization options: size: 16, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:39.058794   356 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 6] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.059075   356 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 92 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 92
I1007 01:51:39.059085   356 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 92 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.059564   356 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 101 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 101
I1007 01:51:39.059576   356 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 101 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.060068   356 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 112 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 112
I1007 01:51:39.060079   356 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 112 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.060411   356 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 116 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 116
I1007 01:51:39.060421   356 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 116 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.060761   356 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 120 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 120
I1007 01:51:39.060771   356 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 120 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.061183   356 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 127 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 127
I1007 01:51:39.061192   356 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 127 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.065809   352 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 91 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 91
I1007 01:51:39.065846   352 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 91 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.066383   352 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 97 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 97
I1007 01:51:39.066396   352 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 97 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.067019   352 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 111 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 111
I1007 01:51:39.067034   352 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 111 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.067409   352 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 115 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 115
I1007 01:51:39.067421   352 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 115 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.067768   352 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 119 Rank 2] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 119
I1007 01:51:39.067781   352 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 119 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.068261   352 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 127 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 127
I1007 01:51:39.068274   352 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 127 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.070390   357 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 15 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 15
I1007 01:51:39.070408   357 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 15 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.070684   352 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 2] ProcessGroupNCCL initialization options: size: 16, global rank: 2, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:39.070703   352 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 2] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.071090   356 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 6] ProcessGroupNCCL initialization options: size: 16, global rank: 6, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:39.071103   356 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 6] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.074731   353 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 15 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 15
I1007 01:51:39.074764   353 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 15 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.077610   353 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 20 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 20
I1007 01:51:39.077634   353 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 20 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.077875   357 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 24 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 24
I1007 01:51:39.077891   357 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 24 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.078333   353 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 33 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 33
I1007 01:51:39.078346   353 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 33 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.078414   357 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 34 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 34
I1007 01:51:39.078425   357 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 34 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
I1007 01:51:39.078797   353 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 37 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 37
I1007 01:51:39.078810   353 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 37 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.078840   357 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 38 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 38
I1007 01:51:39.078851   357 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 38 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.079574   353 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 50 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 50
I1007 01:51:39.079587   353 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 50 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.079780   357 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 62 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 62
I1007 01:51:39.079792   357 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 62 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.079815   353 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 51 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 51
I1007 01:51:39.079826   353 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 51 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.080045   357 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 63 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 63
I1007 01:51:39.080056   357 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 63 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.080054   353 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 52 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 52
I1007 01:51:39.080062   353 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 52 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.080259   357 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 64 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 64
I1007 01:51:39.080271   357 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 64 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
I1007 01:51:39.081138   357 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 7] ProcessGroupNCCL initialization options: size: 16, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:39.081151   357 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 7] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
I1007 01:51:39.081382   353 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 3] ProcessGroupNCCL initialization options: size: 16, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:39.081395   353 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.081404   357 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 7] ProcessGroupNCCL initialization options: size: 16, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:39.081415   357 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 7] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Using network Socket
I1007 01:51:39.081701   353 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 3] ProcessGroupNCCL initialization options: size: 16, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:39.081712   353 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.081727   357 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 92 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 92
I1007 01:51:39.081738   357 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 92 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.082011   353 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 91 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 91
I1007 01:51:39.082022   353 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 91 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
I1007 01:51:39.082262   357 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 102 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 102
I1007 01:51:39.082274   357 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 102 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.082556   353 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 98 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 98
I1007 01:51:39.082569   353 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 98 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.082791   357 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 112 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 112
I1007 01:51:39.082804   357 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 112 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
I1007 01:51:39.083191   357 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 116 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 116
I1007 01:51:39.083204   357 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 116 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.083214   353 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 111 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 111
I1007 01:51:39.083227   353 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 111 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Using network Socket
I1007 01:51:39.083544   357 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 120 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 120
I1007 01:51:39.083556   357 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 120 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.083607   353 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 115 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 115
I1007 01:51:39.083619   353 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 115 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.084014   353 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 119 Rank 3] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 119
I1007 01:51:39.084026   353 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 119 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.084069   357 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 129 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 129
I1007 01:51:39.084081   357 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 129 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.084596   353 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 129 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 129
I1007 01:51:39.084609   353 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 129 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.091698   353 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 3] ProcessGroupNCCL initialization options: size: 16, global rank: 3, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:39.091710   353 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 3] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:39.091869   357 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 7] ProcessGroupNCCL initialization options: size: 16, global rank: 7, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:39.091894   357 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 7] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
W1007 01:51:39.115911   356 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 6]  using GPU 6 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:39.119724   356 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 6] Using non-blocking mode: 0
I1007 01:51:39.119930   356 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 6] ProcessGroupNCCL broadcast unique ID through store took 0.188902 ms
I1007 01:51:39.119969   356 NCCLUtils.cpp:75] Rank 6: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Comm config Blocking set to 1
W1007 01:51:39.126750   352 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 2]  using GPU 2 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:39.130767   352 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 2] Using non-blocking mode: 0
I1007 01:51:39.130971   352 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 0.172862 ms
I1007 01:51:39.131078   352 NCCLUtils.cpp:75] Rank 2: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
W1007 01:51:39.152560   353 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 3]  using GPU 3 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:39.156452   353 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 3] Using non-blocking mode: 0
I1007 01:51:39.156677   353 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 0.195332 ms
I1007 01:51:39.156757   353 NCCLUtils.cpp:75] Rank 3: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Comm config Blocking set to 1
W1007 01:51:39.168853   357 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 7]  using GPU 7 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:39.173133   357 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 7] Using non-blocking mode: 0
I1007 01:51:39.173256   357 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 7] ProcessGroupNCCL broadcast unique ID through store took 0.103701 ms
I1007 01:51:39.173295   357 NCCLUtils.cpp:75] Rank 7: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
WARNING: Logging before InitGoogleLogging() is written to STDERR
I1007 01:51:39.214690   355 ProcessGroupNCCL.cpp:1007] [PG ID 0 PG GUID 0 Rank 5] ProcessGroupNCCL initialization options: size: 16, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 0
I1007 01:51:39.214726   355 ProcessGroupNCCL.cpp:1016] [PG ID 0 PG GUID 0 Rank 5] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
I1007 01:51:39.216075   355 ProcessGroupNCCL.cpp:1007] [PG ID 1 PG GUID 3 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 3
I1007 01:51:39.216094   355 ProcessGroupNCCL.cpp:1016] [PG ID 1 PG GUID 3 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:51:46.862443   351 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 11 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 11
I1007 01:51:46.862468   351 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 11 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.048981   355 ProcessGroupNCCL.cpp:1007] [PG ID 2 PG GUID 11 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 11
I1007 01:51:47.049007   355 ProcessGroupNCCL.cpp:1016] [PG ID 2 PG GUID 11 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.053117   351 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 18 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 18
I1007 01:51:47.053131   351 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 18 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.053782   351 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 33 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 33
I1007 01:51:47.053792   351 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 33 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.054157   351 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 37 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 37
I1007 01:51:47.054167   351 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 37 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.054668   351 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 44 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 44
I1007 01:51:47.054679   351 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 44 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.054876   351 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 45 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 45
I1007 01:51:47.054886   351 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 45 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.055071   351 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 46 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 46
I1007 01:51:47.055080   351 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 46 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.056360   351 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 1] ProcessGroupNCCL initialization options: size: 16, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:47.056370   351 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.056613   351 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 1] ProcessGroupNCCL initialization options: size: 16, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:47.056622   351 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.056882   351 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 91 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 91
I1007 01:51:47.056891   351 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 91 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.057281   351 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 96 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 96
I1007 01:51:47.057291   351 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 96 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.057891   351 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 111 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 111
I1007 01:51:47.057901   351 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 111 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.058252   351 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 115 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 115
I1007 01:51:47.058262   351 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 115 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.058599   351 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 119 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 119
I1007 01:51:47.058609   351 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 119 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.059012   351 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 125 Rank 0] ProcessGroupNCCL initialization options: size: 4, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 125
I1007 01:51:47.059023   351 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 125 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.508191   355 ProcessGroupNCCL.cpp:1007] [PG ID 3 PG GUID 22 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 22
I1007 01:51:47.508211   355 ProcessGroupNCCL.cpp:1016] [PG ID 3 PG GUID 22 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.508814   355 ProcessGroupNCCL.cpp:1007] [PG ID 4 PG GUID 34 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 34
I1007 01:51:47.508826   355 ProcessGroupNCCL.cpp:1016] [PG ID 4 PG GUID 34 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.509253   355 ProcessGroupNCCL.cpp:1007] [PG ID 5 PG GUID 38 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 38
I1007 01:51:47.509265   355 ProcessGroupNCCL.cpp:1016] [PG ID 5 PG GUID 38 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.510044   355 ProcessGroupNCCL.cpp:1007] [PG ID 6 PG GUID 56 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 56
I1007 01:51:47.510056   355 ProcessGroupNCCL.cpp:1016] [PG ID 6 PG GUID 56 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.510294   355 ProcessGroupNCCL.cpp:1007] [PG ID 7 PG GUID 57 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 57
I1007 01:51:47.510305   355 ProcessGroupNCCL.cpp:1016] [PG ID 7 PG GUID 57 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.510535   355 ProcessGroupNCCL.cpp:1007] [PG ID 8 PG GUID 58 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 58
I1007 01:51:47.510555   355 ProcessGroupNCCL.cpp:1016] [PG ID 8 PG GUID 58 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.511626   355 ProcessGroupNCCL.cpp:1007] [PG ID 9 PG GUID 89 Rank 5] ProcessGroupNCCL initialization options: size: 16, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 89
I1007 01:51:47.511638   355 ProcessGroupNCCL.cpp:1016] [PG ID 9 PG GUID 89 Rank 5] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.511927   355 ProcessGroupNCCL.cpp:1007] [PG ID 10 PG GUID 90 Rank 5] ProcessGroupNCCL initialization options: size: 16, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 90
I1007 01:51:47.511938   355 ProcessGroupNCCL.cpp:1016] [PG ID 10 PG GUID 90 Rank 5] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.512228   355 ProcessGroupNCCL.cpp:1007] [PG ID 11 PG GUID 92 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 92
I1007 01:51:47.512239   355 ProcessGroupNCCL.cpp:1016] [PG ID 11 PG GUID 92 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.512715   355 ProcessGroupNCCL.cpp:1007] [PG ID 12 PG GUID 100 Rank 0] ProcessGroupNCCL initialization options: size: 1, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 100
I1007 01:51:47.512727   355 ProcessGroupNCCL.cpp:1016] [PG ID 12 PG GUID 100 Rank 0] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.513254   355 ProcessGroupNCCL.cpp:1007] [PG ID 13 PG GUID 112 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 112
I1007 01:51:47.513266   355 ProcessGroupNCCL.cpp:1016] [PG ID 13 PG GUID 112 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.513603   355 ProcessGroupNCCL.cpp:1007] [PG ID 14 PG GUID 116 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 116
I1007 01:51:47.513614   355 ProcessGroupNCCL.cpp:1016] [PG ID 14 PG GUID 116 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.513993   355 ProcessGroupNCCL.cpp:1007] [PG ID 15 PG GUID 120 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 120
I1007 01:51:47.514005   355 ProcessGroupNCCL.cpp:1016] [PG ID 15 PG GUID 120 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.514407   355 ProcessGroupNCCL.cpp:1007] [PG ID 16 PG GUID 125 Rank 1] ProcessGroupNCCL initialization options: size: 4, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 125
I1007 01:51:47.514420   355 ProcessGroupNCCL.cpp:1016] [PG ID 16 PG GUID 125 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.520927   351 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 1] ProcessGroupNCCL initialization options: size: 16, global rank: 1, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:47.520948   351 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 1] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
I1007 01:51:47.521040   355 ProcessGroupNCCL.cpp:1007] [PG ID 17 PG GUID 131 Rank 5] ProcessGroupNCCL initialization options: size: 16, global rank: 5, TIMEOUT(ms): 1200000, USE_HIGH_PRIORITY_STREAM: 0, SPLIT_FROM: 0, SPLIT_COLOR: -2, PG Name: 131
I1007 01:51:47.521070   355 ProcessGroupNCCL.cpp:1016] [PG ID 17 PG GUID 131 Rank 5] ProcessGroupNCCL environments: NCCL version: 2.22.3, TORCH_NCCL_ASYNC_ERROR_HANDLING: 1, TORCH_NCCL_DUMP_ON_TIMEOUT: 1, TORCH_NCCL_PROPAGATE_ERROR: 0, TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: 60000, TORCH_NCCL_DESYNC_DEBUG: 0, TORCH_NCCL_ENABLE_TIMING: 0, TORCH_NCCL_BLOCKING_WAIT: 0, TORCH_DISTRIBUTED_DEBUG: OFF, TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: 0, TORCH_NCCL_ENABLE_MONITORING: 1, TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: 480, TORCH_NCCL_TRACE_BUFFER_SIZE: 2000, TORCH_NCCL_COORD_CHECK_MILSEC: 1000, TORCH_NCCL_NAN_CHECK: 0, TORCH_NCCL_CUDA_EVENT_CACHE: 1, TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: 1
W1007 01:51:47.592487   351 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:47.596807   351 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 1] Using non-blocking mode: 0
I1007 01:51:47.597057   351 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 0.223762 ms
I1007 01:51:47.597169   351 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Comm config Blocking set to 1
W1007 01:51:47.599092   355 ProcessGroupNCCL.cpp:4834] [PG ID 0 PG GUID 0 Rank 5]  using GPU 5 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
I1007 01:51:47.603317   355 ProcessGroupNCCL.cpp:1107] [PG ID 0 PG GUID 0(default_pg) Rank 5] Using non-blocking mode: 0
I1007 01:51:47.603417   355 ProcessGroupNCCL.cpp:2857] [PG ID 0 PG GUID 0(default_pg) Rank 5] ProcessGroupNCCL broadcast unique ID through store took 0.081801 ms
I1007 01:51:47.603443   355 NCCLUtils.cpp:75] Rank 5: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO ROCr version 1.11
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Dmabuf feature disabled without NCCL_DMABUF_ENABLE=1
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Disabled GDRCopy equivalent memory allocation on gfx936 due to GPU architecture
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Bootstrap : Using eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO RCCL version : 2.22.3-HEAD:19c80f7
HIP version  : 6.3.26102-a22a51fa
ROCm version : 6.3.3.0-0-c1d64df
Hostname     : vcjob-2107528843008176130-abc12335-66689-worker-0
Librccl path : /opt/dtk/lib/librccl.so.1
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO NET/Plugin: Could not find: librccl-net.so. Using internal network plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO NCCL_IB_DISABLE set by environment to 0.
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO NET/IB : No device found.
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO NET/Socket : Using [0]eth0:172.31.50.213<0>
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO ncclCommInitRank comm 0x563be9eb1500 rank 5 nranks 16 cudaDev 5 nvmlDev 5 busId b5000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO ncclCommInitRank comm 0x55ab29eee860 rank 7 nranks 16 cudaDev 7 nvmlDev 7 busId f5000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO ncclCommInitRank comm 0x5612ac5dda10 rank 6 nranks 16 cudaDev 6 nvmlDev 6 busId d5000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO rank:5 get match hsa dev domain:0x0 bdf:0xb500 rank busId:0x741376
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO ncclCommInitRank comm 0x563173ca4940 rank 4 nranks 16 cudaDev 4 nvmlDev 4 busId 85000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO ncclCommInitRank comm 0x563d86ba0a00 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 55000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO ncclCommInitRank comm 0x5586cb640110 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 36000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO rank:7 get match hsa dev domain:0x0 bdf:0xf500 rank busId:0x1003520
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO ncclCommInitRank comm 0x560251f699b0 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 77000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0x3600 rank busId:0x221184
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO ncclCommInitRank comm 0x55692431bdd0 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 9000 commId 0x3711be97a5ee2472 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO rank:2 get match hsa dev domain:0x0 bdf:0x5500 rank busId:0x348160
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO rank:6 get match hsa dev domain:0x0 bdf:0xd500 rank busId:0x872448
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO rank:4 get match hsa dev domain:0x0 bdf:0x8500 rank busId:0x544768
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x900 rank busId:0x36864
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO rank:3 get match hsa dev domain:0x0 bdf:0x7700 rank busId:0x487424
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1839 NCCL WARN Unable to obtain the pciPath[NULL] or name[eth0] attributes from NetProperties.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Setting affinity for GPU 3 to ffff0000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Rank:3 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Transport init rank 3 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Setting affinity for GPU 7 to ffff0000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Rank:7 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Transport init rank 7 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Setting affinity for GPU 0 to ffff
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Setting affinity for GPU 5 to ffff0000,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Rank:5 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Transport init rank 5 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Setting affinity for GPU 1 to ffff0000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Setting affinity for GPU 6 to ffff,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Rank:6 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Transport init rank 6 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Setting affinity for GPU 4 to ffff,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Rank:4 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Transport init rank 4 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Setting affinity for GPU 2 to ffff,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Rank:2 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Transport init rank 2 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO comm 0x5612ac5dda10 rank 6 nRanks 16 nNodes 2 localRanks 8 localRank 6 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO comm 0x563be9eb1500 rank 5 nRanks 16 nNodes 2 localRanks 8 localRank 5 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO comm 0x563173ca4940 rank 4 nRanks 16 nNodes 2 localRanks 8 localRank 4 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO comm 0x560251f699b0 rank 3 nRanks 16 nNodes 2 localRanks 8 localRank 3 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Trees [0] 7/-1/-1->6->5 [1] 7/-1/-1->6->5 [2] 7/-1/-1->6->5 [3] 7/-1/-1->6->5 [4] 7/-1/-1->6->5 [5] 7/-1/-1->6->5 [6] 7/-1/-1->6->5 [7] 7/-1/-1->6->5 [8] 7/-1/-1->6->5 [9] 7/-1/-1->6->5 [10] 7/-1/-1->6->5 [11] 7/-1/-1->6->5 [12] 7/-1/-1->6->5 [13] 7/-1/-1->6->5 [14] 7/-1/-1->6->5 [15] 7/-1/-1->6->5 [16] 7/-1/-1->6->5 [17] 7/-1/-1->6->5 [18] 7/-1/-1->6->5 [19] 7/-1/-1->6->5 [20] 7/-1/-1->6->5 [21] 7/-1/-1->6->5 [22] 7/-1/-1->6->5 [23] 7/-1/-1->6->5 [24] 7/-1/-1->6->5 [25] 7/-1/-1->6->5 [26] 7/-1/-1->6->5 [27] 7/-1/-1->6->5 [28] 7/-1/-1->6->5 [29] 7/-1/-1->6->5 [30] 7/-1/-1->6->5 [31] 7/-1/-1->6->5 comm 0x5612ac5dda10 nRanks 16 busId d5000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO comm 0x563d86ba0a00 rank 2 nRanks 16 nNodes 2 localRanks 8 localRank 2 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Trees [0] 6/-1/-1->5->4 [1] 6/-1/-1->5->4 [2] 6/-1/-1->5->4 [3] 6/-1/-1->5->4 [4] 6/-1/-1->5->4 [5] 6/-1/-1->5->4 [6] 6/-1/-1->5->4 [7] 6/-1/-1->5->4 [8] 6/-1/-1->5->4 [9] 6/-1/-1->5->4 [10] 6/-1/-1->5->4 [11] 6/-1/-1->5->4 [12] 6/-1/-1->5->4 [13] 6/-1/-1->5->4 [14] 6/-1/-1->5->4 [15] 6/-1/-1->5->4 [16] 6/-1/-1->5->4 [17] 6/-1/-1->5->4 [18] 6/-1/-1->5->4 [19] 6/-1/-1->5->4 [20] 6/-1/-1->5->4 [21] 6/-1/-1->5->4 [22] 6/-1/-1->5->4 [23] 6/-1/-1->5->4 [24] 6/-1/-1->5->4 [25] 6/-1/-1->5->4 [26] 6/-1/-1->5->4 [27] 6/-1/-1->5->4 [28] 6/-1/-1->5->4 [29] 6/-1/-1->5->4 [30] 6/-1/-1->5->4 [31] 6/-1/-1->5->4 comm 0x563be9eb1500 nRanks 16 busId b5000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO comm 0x5586cb640110 rank 1 nRanks 16 nNodes 2 localRanks 8 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Trees [0] 5/-1/-1->4->3 [1] 5/-1/-1->4->3 [2] 5/-1/-1->4->3 [3] 5/-1/-1->4->3 [4] 5/-1/-1->4->3 [5] 5/-1/-1->4->3 [6] 5/-1/-1->4->3 [7] 5/-1/-1->4->3 [8] 5/-1/-1->4->3 [9] 5/-1/-1->4->3 [10] 5/-1/-1->4->3 [11] 5/-1/-1->4->3 [12] 5/-1/-1->4->3 [13] 5/-1/-1->4->3 [14] 5/-1/-1->4->3 [15] 5/-1/-1->4->3 [16] 5/-1/-1->4->3 [17] 5/-1/-1->4->3 [18] 5/-1/-1->4->3 [19] 5/-1/-1->4->3 [20] 5/-1/-1->4->3 [21] 5/-1/-1->4->3 [22] 5/-1/-1->4->3 [23] 5/-1/-1->4->3 [24] 5/-1/-1->4->3 [25] 5/-1/-1->4->3 [26] 5/-1/-1->4->3 [27] 5/-1/-1->4->3 [28] 5/-1/-1->4->3 [29] 5/-1/-1->4->3 [30] 5/-1/-1->4->3 [31] 5/-1/-1->4->3 comm 0x563173ca4940 nRanks 16 busId 85000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO comm 0x55692431bdd0 rank 0 nRanks 16 nNodes 2 localRanks 8 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Trees [0] 4/-1/-1->3->2 [1] 4/-1/-1->3->2 [2] 4/-1/-1->3->2 [3] 4/-1/-1->3->2 [4] 4/-1/-1->3->2 [5] 4/-1/-1->3->2 [6] 4/-1/-1->3->2 [7] 4/-1/-1->3->2 [8] 4/-1/-1->3->2 [9] 4/-1/-1->3->2 [10] 4/-1/-1->3->2 [11] 4/-1/-1->3->2 [12] 4/-1/-1->3->2 [13] 4/-1/-1->3->2 [14] 4/-1/-1->3->2 [15] 4/-1/-1->3->2 [16] 4/-1/-1->3->2 [17] 4/-1/-1->3->2 [18] 4/-1/-1->3->2 [19] 4/-1/-1->3->2 [20] 4/-1/-1->3->2 [21] 4/-1/-1->3->2 [22] 4/-1/-1->3->2 [23] 4/-1/-1->3->2 [24] 4/-1/-1->3->2 [25] 4/-1/-1->3->2 [26] 4/-1/-1->3->2 [27] 4/-1/-1->3->2 [28] 4/-1/-1->3->2 [29] 4/-1/-1->3->2 [30] 4/-1/-1->3->2 [31] 4/-1/-1->3->2 comm 0x560251f699b0 nRanks 16 busId 77000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO comm 0x55ab29eee860 rank 7 nRanks 16 nNodes 2 localRanks 8 localRank 7 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->1 [7] 3/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->1 [11] 3/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->1 [15] 3/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->1 [19] 3/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->1 [23] 3/-1/-1->2->1 [24] 3/-1/-1->2->1 [25] 3/-1/-1->2->1 [26] 3/-1/-1->2->1 [27] 3/-1/-1->2->1 [28] 3/-1/-1->2->1 [29] 3/-1/-1->2->1 [30] 3/-1/-1->2->1 [31] 3/-1/-1->2->1 comm 0x563d86ba0a00 nRanks 16 busId 55000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 00/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->0 [6] 2/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->0 [10] 2/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->0 [14] 2/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->0 [18] 2/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->0 [22] 2/-1/-1->1->0 [23] 2/-1/-1->1->0 [24] 2/-1/-1->1->0 [25] 2/-1/-1->1->0 [26] 2/-1/-1->1->0 [27] 2/-1/-1->1->0 [28] 2/-1/-1->1->0 [29] 2/-1/-1->1->0 [30] 2/-1/-1->1->0 [31] 2/-1/-1->1->0 comm 0x5586cb640110 nRanks 16 busId 36000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 01/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 02/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 03/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 04/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 05/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 06/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 07/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 08/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 09/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 10/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 11/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 12/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Trees [0] -1/-1/-1->7->6 [1] -1/-1/-1->7->6 [2] -1/-1/-1->7->6 [3] -1/-1/-1->7->6 [4] -1/-1/-1->7->6 [5] -1/-1/-1->7->6 [6] -1/-1/-1->7->6 [7] -1/-1/-1->7->6 [8] -1/-1/-1->7->6 [9] -1/-1/-1->7->6 [10] -1/-1/-1->7->6 [11] -1/-1/-1->7->6 [12] -1/-1/-1->7->6 [13] -1/-1/-1->7->6 [14] -1/-1/-1->7->6 [15] -1/-1/-1->7->6 [16] -1/-1/-1->7->6 [17] -1/-1/-1->7->6 [18] -1/-1/-1->7->6 [19] -1/-1/-1->7->6 [20] -1/-1/-1->7->6 [21] -1/-1/-1->7->6 [22] -1/-1/-1->7->6 [23] -1/-1/-1->7->6 [24] -1/-1/-1->7->6 [25] -1/-1/-1->7->6 [26] -1/-1/-1->7->6 [27] -1/-1/-1->7->6 [28] -1/-1/-1->7->6 [29] -1/-1/-1->7->6 [30] -1/-1/-1->7->6 [31] -1/-1/-1->7->6 comm 0x55ab29eee860 nRanks 16 busId f5000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 13/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 14/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 15/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 16/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 17/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 18/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 19/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 20/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 21/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 22/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 23/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 24/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 25/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 26/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 27/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 28/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 29/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 30/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 31/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Trees [0] 1/8/-1->0->-1 [1] 1/-1/-1->0->8 [2] 1/8/-1->0->-1 [3] 1/-1/-1->0->8 [4] 1/8/-1->0->-1 [5] 1/-1/-1->0->8 [6] 1/8/-1->0->-1 [7] 1/-1/-1->0->8 [8] 1/8/-1->0->-1 [9] 1/-1/-1->0->8 [10] 1/8/-1->0->-1 [11] 1/-1/-1->0->8 [12] 1/8/-1->0->-1 [13] 1/-1/-1->0->8 [14] 1/8/-1->0->-1 [15] 1/-1/-1->0->8 [16] 1/8/-1->0->-1 [17] 1/-1/-1->0->8 [18] 1/8/-1->0->-1 [19] 1/-1/-1->0->8 [20] 1/8/-1->0->-1 [21] 1/-1/-1->0->8 [22] 1/8/-1->0->-1 [23] 1/-1/-1->0->8 [24] 1/8/-1->0->-1 [25] 1/-1/-1->0->8 [26] 1/8/-1->0->-1 [27] 1/-1/-1->0->8 [28] 1/8/-1->0->-1 [29] 1/-1/-1->0->8 [30] 1/8/-1->0->-1 [31] 1/-1/-1->0->8 comm 0x55692431bdd0 nRanks 16 busId 9000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 00/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 00/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 01/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 02/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 03/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 04/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 05/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 06/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 07/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 08/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 09/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 10/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 11/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 12/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 13/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 14/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 15/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 16/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 17/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 18/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 19/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 20/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 21/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 22/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 23/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 24/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 25/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 26/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 27/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 28/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 29/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 30/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 31/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 00/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 01/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 02/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 00/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 03/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 01/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 00/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 00/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 04/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 02/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 01/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 01/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 05/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 00/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 03/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 02/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 06/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 02/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 04/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 01/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 03/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 07/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 03/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 05/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 02/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 04/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 08/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 06/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 04/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 03/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 05/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 09/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 07/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 04/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 05/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 06/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 10/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 08/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 05/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 07/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 06/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 11/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 09/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 08/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 06/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 12/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 07/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 10/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 09/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 07/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 13/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 08/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 11/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 10/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 08/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 14/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 12/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 09/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 11/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 09/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 15/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 13/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 12/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 10/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 01/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 16/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 10/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 02/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 03/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 14/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 13/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 04/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 05/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 11/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 06/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 17/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 07/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 11/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 08/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 14/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 09/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 10/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 11/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 18/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 12/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 12/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 13/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 12/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 15/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 14/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 15/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 15/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 16/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 19/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 17/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 18/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 13/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 19/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 13/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 16/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 16/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 20/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 21/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 22/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 20/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 23/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 24/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 25/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 26/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 17/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 17/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 14/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 14/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 27/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 21/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 28/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 29/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 30/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 31/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 00/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 18/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 18/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 15/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 15/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 22/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 19/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 19/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 16/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 23/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 16/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 20/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 20/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 24/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 17/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 17/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 21/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 21/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 25/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 18/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 18/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 22/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 22/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 26/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 19/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 19/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 23/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 23/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 27/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 20/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 24/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 24/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 28/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 20/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 21/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 25/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 25/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 29/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 21/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 22/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 26/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 26/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 30/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 22/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 23/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 27/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 27/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 31/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 23/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 28/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 24/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 28/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 24/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 29/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 29/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 25/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 30/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 25/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 30/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 26/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 31/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 31/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 26/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 27/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 28/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 27/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 29/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 28/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 30/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 29/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 31/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 30/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 31/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 01/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 02/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 03/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 04/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 05/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 06/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 07/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 08/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 09/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 10/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 11/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 12/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 13/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 14/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 15/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 16/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 17/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 18/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 19/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 20/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 21/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 22/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 23/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 24/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 25/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 26/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 27/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 28/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 29/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 30/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 31/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 00/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 01/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 02/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 03/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 04/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 05/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 06/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 07/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 08/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 09/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 10/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 11/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 12/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 13/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 14/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 15/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 16/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 17/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 18/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 19/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 20/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 21/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 22/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 23/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 24/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 25/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 00/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 26/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 01/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 27/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 02/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 28/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 03/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 29/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 04/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 30/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 05/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Channel 31/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab29eee860 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 06/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 07/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 08/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 09/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 10/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 11/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 12/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 13/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 14/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 15/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 16/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 17/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 18/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 19/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 20/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 21/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 22/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 23/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 24/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 25/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 26/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 27/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 28/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 29/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 30/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Channel 31/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x563173ca4940 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 00/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 01/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 02/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 03/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 04/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 05/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 06/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 07/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 00/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 08/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 09/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 10/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 11/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 12/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 13/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 14/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 15/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 16/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 17/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 18/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 19/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 20/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 21/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 22/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 23/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 24/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 25/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 26/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 27/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 28/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 29/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 30/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Channel 31/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563be9eb1500 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 00/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 01/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 02/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 01/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 03/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 02/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 04/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 03/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 05/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 04/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 06/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 05/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 06/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 07/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 07/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 08/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 08/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 09/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 09/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 10/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 10/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 11/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 11/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 12/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 12/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 13/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 13/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 14/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 14/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 15/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 15/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 16/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 16/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 17/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 18/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 17/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 19/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 18/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 20/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 19/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 21/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 20/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 22/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 21/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 23/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 24/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 22/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 25/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 23/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 26/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 24/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 27/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 25/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 28/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 26/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 29/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 30/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 27/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 31/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 28/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 29/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 30/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Channel 31/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x560251f699b0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 00/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 01/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 02/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 03/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 04/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 05/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 06/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 07/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 08/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 09/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 10/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 11/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 12/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 00/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 13/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 01/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 14/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 02/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 15/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 03/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 04/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 16/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 05/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 17/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 06/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 18/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 07/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 19/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 00/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 08/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 20/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 01/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 09/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 21/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 02/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 10/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 03/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 22/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 11/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 04/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 12/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 23/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 05/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 13/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 24/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 06/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 14/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 25/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 07/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 15/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 08/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 26/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 16/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 09/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 27/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 17/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 10/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 28/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 18/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 11/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 29/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 19/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 12/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 20/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 30/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 13/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 21/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 14/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Channel 31/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612ac5dda10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 22/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 15/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 23/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 16/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 17/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 24/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 18/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 25/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 19/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 26/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 20/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 27/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 21/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 28/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 22/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 29/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 23/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 30/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 24/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 31/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 25/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 26/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 00/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 01/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 02/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 03/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 04/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 27/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 28/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 29/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 30/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Channel 31/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d86ba0a00 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 05/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 06/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 07/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 08/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 00/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 09/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 10/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 11/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 12/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 13/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 14/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 01/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 15/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 16/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 17/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 18/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 19/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 20/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 02/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 21/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 22/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 23/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 24/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 25/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 26/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 03/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 27/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 28/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 29/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 30/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 31/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 04/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 05/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 06/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 07/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 08/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 09/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 10/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 11/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 12/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 13/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 14/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 15/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 16/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 17/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 18/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 19/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 20/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 21/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 22/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 23/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 24/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 25/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 26/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 27/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 28/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 29/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 30/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Channel 31/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cb640110 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 00/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 01/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 02/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 03/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 04/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 05/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 06/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 07/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 08/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 09/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 10/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 11/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 12/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 13/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 14/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 15/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 16/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 17/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 18/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 19/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 20/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 21/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 22/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 23/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 24/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 25/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 26/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 27/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 28/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 29/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 30/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Channel 31/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x55692431bdd0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO ncclCommInitRank comm 0x5612ac5dda10 rank 6 nranks 16 cudaDev 6 nvmlDev 6 busId d5000 commId 0x3711be97a5ee2472 localSize 480 used 282378608 bytes on core 104 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:746 [6] NCCL INFO Init timings: rank 6 nranks 16 total 10.11 (kernels 0.06, bootstrap 8.61, allgathers 0.00, topo 0.94, graphs 0.00, connections 0.30, rest 0.19)
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO ncclCommInitRank comm 0x563d86ba0a00 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 55000 commId 0x3711be97a5ee2472 localSize 480 used 282378608 bytes on core 35 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:352:748 [2] NCCL INFO Init timings: rank 2 nranks 16 total 10.10 (kernels 0.05, bootstrap 8.61, allgathers 0.00, topo 0.94, graphs 0.00, connections 0.37, rest 0.12)
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO ncclCommInitRank comm 0x563be9eb1500 rank 5 nranks 16 cudaDev 5 nvmlDev 5 busId b5000 commId 0x3711be97a5ee2472 localSize 480 used 282378608 bytes on core 114 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:355:848 [5] NCCL INFO Init timings: rank 5 nranks 16 total 1.62 (kernels 0.05, bootstrap 0.13, allgathers 0.02, topo 0.93, graphs 0.00, connections 0.30, rest 0.19)
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO ncclCommInitRank comm 0x560251f699b0 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 77000 commId 0x3711be97a5ee2472 localSize 480 used 282378608 bytes on core 26 - Init COMPLETE
I1007 01:51:49.227638   356 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 6] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:353:749 [3] NCCL INFO Init timings: rank 3 nranks 16 total 10.07 (kernels 0.05, bootstrap 8.58, allgathers 0.06, topo 0.89, graphs 0.00, connections 0.31, rest 0.18)
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO ncclCommInitRank comm 0x563173ca4940 rank 4 nranks 16 cudaDev 4 nvmlDev 4 busId 85000 commId 0x3711be97a5ee2472 localSize 480 used 282378608 bytes on core 69 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:354:582 [4] NCCL INFO Init timings: rank 4 nranks 16 total 10.20 (kernels 0.05, bootstrap 8.71, allgathers 0.00, topo 0.94, graphs 0.00, connections 0.31, rest 0.18)
I1007 01:51:49.227739   352 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 2] NCCL_DEBUG: INFO
I1007 01:51:49.227767   355 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 5] NCCL_DEBUG: INFO
I1007 01:51:49.227902   353 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 3] NCCL_DEBUG: INFO
I1007 01:51:49.227950   354 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 4] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO ncclCommInitRank comm 0x55692431bdd0 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 9000 commId 0x3711be97a5ee2472 localSize 480 used 225624432 bytes on core 8 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:350:581 [0] NCCL INFO Init timings: rank 0 nranks 16 total 10.20 (kernels 0.05, bootstrap 8.71, allgathers 0.02, topo 0.92, graphs 0.00, connections 0.47, rest 0.02)
I1007 01:51:49.228865   350 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 0] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO ncclCommInitRank comm 0x5586cb640110 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 36000 commId 0x3711be97a5ee2472 localSize 480 used 282378608 bytes on core 53 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:351:847 [1] NCCL INFO Init timings: rank 1 nranks 16 total 1.63 (kernels 0.06, bootstrap 0.13, allgathers 0.01, topo 0.94, graphs 0.00, connections 0.47, rest 0.02)
I1007 01:51:49.229192   351 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 1] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO TUNER/Plugin: Could not find: librccl-tuner.so librccl-net.so. Using internal tuner plugin.
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO ncclCommInitRank comm 0x55ab29eee860 rank 7 nranks 16 cudaDev 7 nvmlDev 7 busId f5000 commId 0x3711be97a5ee2472 localSize 480 used 215269744 bytes on core 85 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:357:750 [7] NCCL INFO Init timings: rank 7 nranks 16 total 10.06 (kernels 0.05, bootstrap 8.57, allgathers 0.03, topo 0.92, graphs 0.00, connections 0.30, rest 0.19)
I1007 01:51:49.229965   357 ProcessGroupNCCL.cpp:2890] [PG ID 0 PG GUID 0(default_pg) Rank 7] NCCL_DEBUG: INFO
time to initialize megatron (seconds): 15.742
[after in-process setup and before initialize_megatron] datetime: 2026-10-07 01:51:38.556264 
[after in-job setup and before initialize_megatron] datetime: 2026-10-07 01:51:38.669817 
[before library-setup] datetime: 2026-10-07 01:51:38.018864
[after library-setup] datetime: 2026-10-07 01:51:38.034732
[before megatron-init] datetime: 2026-10-07 01:51:38.556263
[after megatron is initialized] datetime: 2026-10-07 01:51:49.288800 
building GPT model ...
 > number of parameters on (tensor, pipeline) model parallel rank (3, 0): 2012483584
 > number of parameters on (tensor, pipeline) model parallel rank (2, 0): 2012483584
 > number of parameters on (tensor, pipeline) model parallel rank (1, 0): 2012483584
[rank 0] GPTModel(
  (embedding): LanguageModelEmbedding(
    (word_embeddings): VocabParallelEmbedding()
    (embedding_dropout): Dropout(p=0.0, inplace=False)
  )
  (rotary_pos_emb): RotaryEmbedding()
  (decoder): TransformerBlock(
    (layers): ModuleList(
      (0-31): 32 x TransformerLayer(
        (input_layernorm): IdentityOp()
        (self_attention): SelfAttention(
          (core_attention): TEDotProductAttention(
            (flash_attention): FlashAttention()
            (fused_attention): FusedAttention()
            (unfused_attention): UnfusedDotProductAttention(
              (scale_mask_softmax): FusedScaleMaskSoftmax()
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
          )
          (linear_proj): TERowParallelLinear(in_features=1024, out_features=4096, bias=False, TP=4)
          (linear_qkv): TELayerNormColumnParallelLinear(in_features=4096, out_features=3072, bias=False, TP=4)
        )
        (pre_cross_attn_layernorm): IdentityOp()
        (cross_attention): IdentityOp()
        (cross_attn_bda): IdentityFuncOp()
        (pre_mlp_layernorm): IdentityOp()
        (mlp): MLP(
          (linear_fc1): TELayerNormColumnParallelLinear(in_features=4096, out_features=7168, bias=False, TP=4)
          (linear_fc2): TERowParallelLinear(in_features=3584, out_features=4096, bias=False, TP=4)
        )
      )
    )
    (final_layernorm): RMSNorm()
  )
  (output_layer): ColumnParallelLinear(in_features=4096, out_features=32256, bias=False, TP=4)
)
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 2012483584
[after model, optimizer, and learning rate scheduler are built] datetime: 2026-10-07 01:51:49.611561 
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      80
    validation: 0
    test:       0
> building train, validation, and test datasets for GPT ...
[WARNING  | megatron.core.datasets.gpt_dataset]: Unable to save MockGPTDataset indexes because path_to_cache is None
[WARNING  | megatron.core.datasets.gpt_dataset]: Unable to save MockGPTDataset indexes because path_to_cache is None
[WARNING  | megatron.core.datasets.gpt_dataset]: Unable to save MockGPTDataset indexes because path_to_cache is None
> finished creating GPT datasets ...
[after dataloaders are built] datetime: 2026-10-07 01:51:52.074203 
done with setup ...
training ...
Overwriting rerun_state_machine.current_iteration from -1 to 0...
[before the start of training step] datetime: 2026-10-07 01:51:52.087816 
I1007 01:51:52.088724   356 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 2] Using non-blocking mode: 0
I1007 01:51:52.088833   353 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] Using non-blocking mode: 0
I1007 01:51:52.088840   351 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 1] Using non-blocking mode: 0
I1007 01:51:52.088851   357 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] Using non-blocking mode: 0
I1007 01:51:52.088891   352 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 2] Using non-blocking mode: 0
I1007 01:51:52.088974   355 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 1] Using non-blocking mode: 0
I1007 01:51:52.110798   350 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:52.111800   350 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.07108 ms
I1007 01:51:52.111827   350 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:51:52.111918   353 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 23.0589 ms
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:52.111945   353 NCCLUtils.cpp:75] Rank 3: creating NCCL communicator with mode: blocking
I1007 01:51:52.111953   352 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 23.0214 ms
I1007 01:51:52.111943   351 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 23.0689 ms
I1007 01:51:52.111994   352 NCCLUtils.cpp:75] Rank 2: creating NCCL communicator with mode: blocking
I1007 01:51:52.111996   351 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:51:52.125448   354 ProcessGroupNCCL.cpp:1107] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:52.126001   354 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.082461 ms
I1007 01:51:52.126030   354 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:52.126121   355 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 37.1159 ms
I1007 01:51:52.126116   356 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 37.3653 ms
I1007 01:51:52.126143   355 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
I1007 01:51:52.126144   356 NCCLUtils.cpp:75] Rank 2: creating NCCL communicator with mode: blocking
I1007 01:51:52.126139   357 ProcessGroupNCCL.cpp:2857] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 37.2537 ms
I1007 01:51:52.126178   357 NCCLUtils.cpp:75] Rank 3: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO ncclCommInitRank comm 0x563d87d9b710 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 55000 commId 0xc651d053a69719d2 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO ncclCommInitRank comm 0x56025311e9a0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 77000 commId 0xc651d053a69719d2 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO ncclCommInitRank comm 0x5569256efd60 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 9000 commId 0xc651d053a69719d2 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO ncclCommInitRank comm 0x5586cc83a020 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 36000 commId 0xc651d053a69719d2 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO rank:3 get match hsa dev domain:0x0 bdf:0x7700 rank busId:0x487424
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x900 rank busId:0x36864
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO rank:2 get match hsa dev domain:0x0 bdf:0x5500 rank busId:0x348160
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0x3600 rank busId:0x221184
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO ncclCommInitRank comm 0x563175078b60 rank 0 nranks 4 cudaDev 4 nvmlDev 4 busId 85000 commId 0xf8e47c66a2098292 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO ncclCommInitRank comm 0x563beb066d70 rank 1 nranks 4 cudaDev 5 nvmlDev 5 busId b5000 commId 0xf8e47c66a2098292 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO ncclCommInitRank comm 0x55ab2b0f6560 rank 3 nranks 4 cudaDev 7 nvmlDev 7 busId f5000 commId 0xf8e47c66a2098292 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO ncclCommInitRank comm 0x5612ad795510 rank 2 nranks 4 cudaDev 6 nvmlDev 6 busId d5000 commId 0xf8e47c66a2098292 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x8500 rank busId:0x544768
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0xb500 rank busId:0x741376
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO rank:2 get match hsa dev domain:0x0 bdf:0xd500 rank busId:0x872448
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO rank:3 get match hsa dev domain:0x0 bdf:0xf500 rank busId:0x1003520
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Setting affinity for GPU 0 to ffff
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Setting affinity for GPU 2 to ffff,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Rank:2 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Transport init rank 2 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Setting affinity for GPU 1 to ffff0000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Setting affinity for GPU 3 to ffff0000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Rank:3 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Transport init rank 3 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO comm 0x56025311e9a0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO comm 0x5569256efd60 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO comm 0x563d87d9b710 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO comm 0x5586cc83a020 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] 0/-1/-1->3->2 [2] 0/-1/-1->3->2 [3] 0/-1/-1->3->-1 [4] -1/-1/-1->3->2 [5] 0/-1/-1->3->2 [6] 0/-1/-1->3->2 [7] 0/-1/-1->3->-1 [8] -1/-1/-1->3->2 [9] 0/-1/-1->3->2 [10] 0/-1/-1->3->2 [11] 0/-1/-1->3->-1 [12] -1/-1/-1->3->2 [13] 0/-1/-1->3->2 [14] 0/-1/-1->3->2 [15] 0/-1/-1->3->-1 [16] -1/-1/-1->3->2 [17] 0/-1/-1->3->2 [18] 0/-1/-1->3->2 [19] 0/-1/-1->3->-1 [20] -1/-1/-1->3->2 [21] 0/-1/-1->3->2 [22] 0/-1/-1->3->2 [23] 0/-1/-1->3->-1 [24] -1/-1/-1->3->2 [25] 0/-1/-1->3->2 [26] 0/-1/-1->3->2 [27] 0/-1/-1->3->-1 [28] -1/-1/-1->3->2 [29] 0/-1/-1->3->2 [30] 0/-1/-1->3->2 [31] 0/-1/-1->3->-1 comm 0x56025311e9a0 nRanks 04 busId 77000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->-1 [3] -1/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->-1 [7] -1/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->-1 [11] -1/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->-1 [15] -1/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->-1 [19] -1/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->-1 [23] -1/-1/-1->2->1 [24] 3/-1/-1->2->1 [25] 3/-1/-1->2->1 [26] 3/-1/-1->2->-1 [27] -1/-1/-1->2->1 [28] 3/-1/-1->2->1 [29] 3/-1/-1->2->1 [30] 3/-1/-1->2->-1 [31] -1/-1/-1->2->1 comm 0x563d87d9b710 nRanks 04 busId 55000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->-1 [2] -1/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->-1 [6] -1/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->-1 [10] -1/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->-1 [14] -1/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->-1 [18] -1/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->-1 [22] -1/-1/-1->1->0 [23] 2/-1/-1->1->0 [24] 2/-1/-1->1->0 [25] 2/-1/-1->1->-1 [26] -1/-1/-1->1->0 [27] 2/-1/-1->1->0 [28] 2/-1/-1->1->0 [29] 2/-1/-1->1->-1 [30] -1/-1/-1->1->0 [31] 2/-1/-1->1->0 comm 0x5586cc83a020 nRanks 04 busId 36000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->3 [2] 1/-1/-1->0->3 [3] 1/-1/-1->0->3 [4] 1/-1/-1->0->-1 [5] -1/-1/-1->0->3 [6] 1/-1/-1->0->3 [7] 1/-1/-1->0->3 [8] 1/-1/-1->0->-1 [9] -1/-1/-1->0->3 [10] 1/-1/-1->0->3 [11] 1/-1/-1->0->3 [12] 1/-1/-1->0->-1 [13] -1/-1/-1->0->3 [14] 1/-1/-1->0->3 [15] 1/-1/-1->0->3 [16] 1/-1/-1->0->-1 [17] -1/-1/-1->0->3 [18] 1/-1/-1->0->3 [19] 1/-1/-1->0->3 [20] 1/-1/-1->0->-1 [21] -1/-1/-1->0->3 [22] 1/-1/-1->0->3 [23] 1/-1/-1->0->3 [24] 1/-1/-1->0->-1 [25] -1/-1/-1->0->3 [26] 1/-1/-1->0->3 [27] 1/-1/-1->0->3 [28] 1/-1/-1->0->-1 [29] -1/-1/-1->0->3 [30] 1/-1/-1->0->3 [31] 1/-1/-1->0->3 comm 0x5569256efd60 nRanks 04 busId 9000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Setting affinity for GPU 6 to ffff,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Rank:2 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Transport init rank 2 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Setting affinity for GPU 4 to ffff,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 00/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 01/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 02/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 03/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 04/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 05/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 00/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 06/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Setting affinity for GPU 7 to ffff0000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 01/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Rank:3 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Transport init rank 3 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 07/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 02/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 08/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 03/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 09/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 04/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 10/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 05/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 11/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 06/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 12/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 07/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 00/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 13/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 08/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 01/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 14/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 09/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 02/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 15/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 10/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 03/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 16/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 11/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 17/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 04/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 12/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Setting affinity for GPU 5 to ffff0000,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 18/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 13/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 05/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 19/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 14/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 06/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 00/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 15/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 20/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 01/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO comm 0x563beb066d70 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO comm 0x55ab2b0f6560 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO comm 0x563175078b60 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->-1 [2] -1/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->-1 [6] -1/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->-1 [10] -1/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->-1 [14] -1/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->-1 [18] -1/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->-1 [22] -1/-1/-1->1->0 [23] 2/-1/-1->1->0 [24] 2/-1/-1->1->0 [25] 2/-1/-1->1->-1 [26] -1/-1/-1->1->0 [27] 2/-1/-1->1->0 [28] 2/-1/-1->1->0 [29] 2/-1/-1->1->-1 [30] -1/-1/-1->1->0 [31] 2/-1/-1->1->0 comm 0x563beb066d70 nRanks 04 busId b5000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] 0/-1/-1->3->2 [2] 0/-1/-1->3->2 [3] 0/-1/-1->3->-1 [4] -1/-1/-1->3->2 [5] 0/-1/-1->3->2 [6] 0/-1/-1->3->2 [7] 0/-1/-1->3->-1 [8] -1/-1/-1->3->2 [9] 0/-1/-1->3->2 [10] 0/-1/-1->3->2 [11] 0/-1/-1->3->-1 [12] -1/-1/-1->3->2 [13] 0/-1/-1->3->2 [14] 0/-1/-1->3->2 [15] 0/-1/-1->3->-1 [16] -1/-1/-1->3->2 [17] 0/-1/-1->3->2 [18] 0/-1/-1->3->2 [19] 0/-1/-1->3->-1 [20] -1/-1/-1->3->2 [21] 0/-1/-1->3->2 [22] 0/-1/-1->3->2 [23] 0/-1/-1->3->-1 [24] -1/-1/-1->3->2 [25] 0/-1/-1->3->2 [26] 0/-1/-1->3->2 [27] 0/-1/-1->3->-1 [28] -1/-1/-1->3->2 [29] 0/-1/-1->3->2 [30] 0/-1/-1->3->2 [31] 0/-1/-1->3->-1 comm 0x55ab2b0f6560 nRanks 04 busId f5000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 07/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO comm 0x5612ad795510 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->-1 [3] -1/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->-1 [7] -1/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->-1 [11] -1/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->-1 [15] -1/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->-1 [19] -1/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->-1 [23] -1/-1/-1->2->1 [24] 3/-1/-1->2->1 [25] 3/-1/-1->2->1 [26] 3/-1/-1->2->-1 [27] -1/-1/-1->2->1 [28] 3/-1/-1->2->1 [29] 3/-1/-1->2->1 [30] 3/-1/-1->2->-1 [31] -1/-1/-1->2->1 comm 0x5612ad795510 nRanks 04 busId d5000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->3 [2] 1/-1/-1->0->3 [3] 1/-1/-1->0->3 [4] 1/-1/-1->0->-1 [5] -1/-1/-1->0->3 [6] 1/-1/-1->0->3 [7] 1/-1/-1->0->3 [8] 1/-1/-1->0->-1 [9] -1/-1/-1->0->3 [10] 1/-1/-1->0->3 [11] 1/-1/-1->0->3 [12] 1/-1/-1->0->-1 [13] -1/-1/-1->0->3 [14] 1/-1/-1->0->3 [15] 1/-1/-1->0->3 [16] 1/-1/-1->0->-1 [17] -1/-1/-1->0->3 [18] 1/-1/-1->0->3 [19] 1/-1/-1->0->3 [20] 1/-1/-1->0->-1 [21] -1/-1/-1->0->3 [22] 1/-1/-1->0->3 [23] 1/-1/-1->0->3 [24] 1/-1/-1->0->-1 [25] -1/-1/-1->0->3 [26] 1/-1/-1->0->3 [27] 1/-1/-1->0->3 [28] 1/-1/-1->0->-1 [29] -1/-1/-1->0->3 [30] 1/-1/-1->0->3 [31] 1/-1/-1->0->3 comm 0x563175078b60 nRanks 04 busId 85000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 16/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 21/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 02/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 08/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 17/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 22/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 03/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 09/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 18/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 23/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 04/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 19/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 10/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 24/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 05/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 20/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 25/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 11/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 06/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 21/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 07/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 26/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 12/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 22/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 08/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 27/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 13/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 23/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 09/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 28/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 14/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 24/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 10/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 29/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 25/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 15/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 11/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 30/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 26/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 16/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 12/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 31/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 27/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 13/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 17/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 28/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 14/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 18/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 29/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 15/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 30/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 19/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 16/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 31/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 20/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 17/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 18/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 21/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 19/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 22/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 20/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 23/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 21/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 24/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 22/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 25/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 23/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 24/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 26/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 25/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 27/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 26/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 28/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 27/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 29/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 28/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 30/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 29/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 31/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 30/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 31/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 00/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 01/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 02/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 03/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 00/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 00/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 04/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 01/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 01/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 05/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 02/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 02/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 06/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 03/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 00/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 03/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 04/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 07/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 01/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 04/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 05/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 08/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 02/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 05/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 06/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 09/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 03/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 06/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 07/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 10/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 04/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 07/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 08/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 11/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 05/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 08/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 09/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 12/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 06/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 09/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 10/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 13/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 07/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 10/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 11/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 14/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 08/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 11/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 12/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 15/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 09/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 12/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 13/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 13/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 16/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 10/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 14/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 14/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 17/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 11/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 15/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 15/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 18/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 12/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 16/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 16/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 19/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 13/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 17/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 17/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 20/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 14/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 18/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 18/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 21/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 15/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 19/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 19/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 22/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 16/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 20/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 20/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 23/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 21/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 17/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 21/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 24/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 25/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 23/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 19/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 23/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 26/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 24/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 20/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 24/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 25/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 27/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 21/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 25/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 01/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 28/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 22/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 26/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 26/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 02/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 29/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 27/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 23/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 27/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 03/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 30/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 00/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 28/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 28/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 24/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 05/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 31/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 29/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 29/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 25/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 01/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 06/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 30/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 30/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 26/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 02/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 00/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 07/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 31/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 31/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 27/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 00/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 02/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 04/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 09/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 28/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 01/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 03/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 05/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 10/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 29/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 03/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 04/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 06/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 11/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 30/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 04/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 06/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 08/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 13/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 31/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 05/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 07/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 09/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 14/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 07/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 08/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 15/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 10/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 08/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 10/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 17/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 12/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 09/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 11/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 18/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 13/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 12/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 11/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 19/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 14/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 14/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 12/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 21/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 16/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 13/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 15/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 22/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 15/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 16/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 17/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 23/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 16/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 18/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 18/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 25/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 17/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 19/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 20/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 26/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 20/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 19/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 27/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 21/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 22/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 20/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 29/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 22/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 23/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 21/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 30/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 24/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 24/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 23/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Channel 31/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x5569256efd60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 25/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 26/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 24/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 26/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 27/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 25/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 28/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 28/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 27/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 29/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 30/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 28/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Channel 31/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586cc83a020 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Channel 30/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x56025311e9a0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 29/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Channel 31/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563d87d9b710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO ncclCommInitRank comm 0x5586cc83a020 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 36000 commId 0xc651d053a69719d2 localSize 480 used 533398752 bytes on core 49 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO ncclCommInitRank comm 0x56025311e9a0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 77000 commId 0xc651d053a69719d2 localSize 480 used 533398752 bytes on core 16 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:351:939 [1] NCCL INFO Init timings: rank 1 nranks 4 total 0.69 (kernels 0.00, bootstrap 0.08, allgathers 0.01, topo 0.44, graphs 0.00, connections 0.16, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO ncclCommInitRank comm 0x5569256efd60 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 9000 commId 0xc651d053a69719d2 localSize 480 used 476644576 bytes on core 0 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:353:938 [3] NCCL INFO Init timings: rank 3 nranks 4 total 0.69 (kernels 0.00, bootstrap 0.08, allgathers 0.00, topo 0.45, graphs 0.00, connections 0.16, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:350:941 [0] NCCL INFO Init timings: rank 0 nranks 4 total 0.69 (kernels 0.00, bootstrap 0.07, allgathers 0.02, topo 0.43, graphs 0.00, connections 0.16, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO ncclCommInitRank comm 0x563d87d9b710 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 55000 commId 0xc651d053a69719d2 localSize 480 used 533398752 bytes on core 32 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:352:940 [2] NCCL INFO Init timings: rank 2 nranks 4 total 0.69 (kernels 0.00, bootstrap 0.08, allgathers 0.02, topo 0.43, graphs 0.00, connections 0.16, rest 0.00)
I1007 01:51:52.805007   351 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 1] NCCL_DEBUG: INFO
I1007 01:51:52.805022   353 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] NCCL_DEBUG: INFO
I1007 01:51:52.805032   352 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 2] NCCL_DEBUG: INFO
I1007 01:51:52.805033   350 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 37(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 00/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 01/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 03/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 00/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 04/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 01/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 02/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 02/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 05/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 00/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 03/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 03/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 07/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 05/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 04/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 01/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 08/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 06/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 06/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 02/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 09/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 07/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 07/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 04/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 11/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 09/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 08/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 05/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 12/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 10/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 10/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 06/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 13/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 11/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 11/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 08/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 15/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 13/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 12/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 09/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 14/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 16/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 14/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 10/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 15/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 17/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 15/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 12/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 17/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 18/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 13/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 19/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 16/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 19/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 14/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 18/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 20/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 21/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 16/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 19/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 21/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 22/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 17/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 20/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 23/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 23/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 18/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 22/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 25/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 24/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 20/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 23/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 26/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 25/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 21/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 27/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 24/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 27/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 22/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 29/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 26/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 28/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 24/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 30/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 27/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 29/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Channel 31/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x563175078b60 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 25/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 28/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Channel 31/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612ad795510 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 26/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 30/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 28/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Channel 31/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563beb066d70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 29/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Channel 30/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab2b0f6560 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO ncclCommInitRank comm 0x563175078b60 rank 0 nranks 4 cudaDev 4 nvmlDev 4 busId 85000 commId 0xf8e47c66a2098292 localSize 480 used 533398752 bytes on core 72 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO ncclCommInitRank comm 0x5612ad795510 rank 2 nranks 4 cudaDev 6 nvmlDev 6 busId d5000 commId 0xf8e47c66a2098292 localSize 480 used 533398752 bytes on core 98 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO ncclCommInitRank comm 0x563beb066d70 rank 1 nranks 4 cudaDev 5 nvmlDev 5 busId b5000 commId 0xf8e47c66a2098292 localSize 480 used 533398752 bytes on core 112 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:354:946 [4] NCCL INFO Init timings: rank 0 nranks 4 total 0.73 (kernels 0.00, bootstrap 0.07, allgathers 0.04, topo 0.45, graphs 0.00, connections 0.16, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:356:944 [6] NCCL INFO Init timings: rank 2 nranks 4 total 0.73 (kernels 0.00, bootstrap 0.07, allgathers 0.05, topo 0.44, graphs 0.00, connections 0.16, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:355:943 [5] NCCL INFO Init timings: rank 1 nranks 4 total 0.73 (kernels 0.00, bootstrap 0.08, allgathers 0.01, topo 0.48, graphs 0.00, connections 0.16, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO ncclCommInitRank comm 0x55ab2b0f6560 rank 3 nranks 4 cudaDev 7 nvmlDev 7 busId f5000 commId 0xf8e47c66a2098292 localSize 480 used 466289888 bytes on core 80 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:357:945 [7] NCCL INFO Init timings: rank 3 nranks 4 total 0.73 (kernels 0.00, bootstrap 0.07, allgathers 0.02, topo 0.47, graphs 0.00, connections 0.16, rest 0.00)
I1007 01:51:52.858491   355 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 1] NCCL_DEBUG: INFO
I1007 01:51:52.858494   356 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 2] NCCL_DEBUG: INFO
I1007 01:51:52.858505   354 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 0] NCCL_DEBUG: INFO
I1007 01:51:52.859542   357 ProcessGroupNCCL.cpp:2890] [PG ID 5 PG GUID 38(TENSOR_MODEL_PARALLEL_GROUP) Rank 3] NCCL_DEBUG: INFO
[WARNING  | megatron.core.rerun_state_machine]: Result validation enabled
I1007 01:51:57.473037   351 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 3(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:57.473506   351 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 3(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.0574 ms
I1007 01:51:57.473532   351 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:57.479694   353 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 7(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Kernel version: 5.15.0-25-generic
I1007 01:51:57.479851   352 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 5(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:51:57.479872   350 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] Using non-blocking mode: 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:57.480175   353 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 7(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.06058 ms
I1007 01:51:57.480198   353 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:57.480283   352 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 5(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.04757 ms
I1007 01:51:57.480306   352 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:51:57.480305   350 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.049851 ms
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
I1007 01:51:57.480329   350 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Kernel version: 5.15.0-25-generic
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:51:57.586733   355 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 3(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] Using non-blocking mode: 0
I1007 01:51:57.586969   355 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 3(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 0.196782 ms
I1007 01:51:57.587024   355 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:57.598058   356 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 5(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] Using non-blocking mode: 0
I1007 01:51:57.598227   356 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 5(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 0.132831 ms
I1007 01:51:57.598261   356 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:51:57.601150   354 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] Using non-blocking mode: 0
I1007 01:51:57.601343   354 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 0.155111 ms
I1007 01:51:57.601368   354 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:51:57.605434   357 ProcessGroupNCCL.cpp:1107] [PG ID 1 PG GUID 7(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] Using non-blocking mode: 0
I1007 01:51:57.605629   357 ProcessGroupNCCL.cpp:2857] [PG ID 1 PG GUID 7(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 0.155231 ms
I1007 01:51:57.605664   357 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO ncclCommInitRank comm 0x5612cbb74d90 rank 1 nranks 4 cudaDev 6 nvmlDev 6 busId d5000 commId 0x5ecafbf2e3c52cd0 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0xd500 rank busId:0x872448
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO ncclCommInitRank comm 0x563da60624e0 rank 0 nranks 4 cudaDev 2 nvmlDev 2 busId 55000 commId 0x5ecafbf2e3c52cd0 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x5500 rank busId:0x348160
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO ncclCommInitRank comm 0x55ab4948db90 rank 1 nranks 4 cudaDev 7 nvmlDev 7 busId f5000 commId 0x47c875b4b070469a - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO ncclCommInitRank comm 0x5602714d6080 rank 0 nranks 4 cudaDev 3 nvmlDev 3 busId 77000 commId 0x47c875b4b070469a - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0xf500 rank busId:0x1003520
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x7700 rank busId:0x487424
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO ncclCommInitRank comm 0x5586eabac5b0 rank 0 nranks 4 cudaDev 1 nvmlDev 1 busId 36000 commId 0x816cb0dbd678753f - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO ncclCommInitRank comm 0x563c0944bcb0 rank 1 nranks 4 cudaDev 5 nvmlDev 5 busId b5000 commId 0x816cb0dbd678753f - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0xb500 rank busId:0x741376
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x3600 rank busId:0x221184
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO ncclCommInitRank comm 0x5631932ed400 rank 1 nranks 4 cudaDev 4 nvmlDev 4 busId 85000 commId 0x6f61729fbd02ce20 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO ncclCommInitRank comm 0x556943960410 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 9000 commId 0x6f61729fbd02ce20 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0x8500 rank busId:0x544768
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x900 rank busId:0x36864
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Setting affinity for GPU 7 to ffff0000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Setting affinity for GPU 3 to ffff0000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO comm 0x5602714d6080 rank 0 nRanks 4 nNodes 2 localRanks 2 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO comm 0x55ab4948db90 rank 1 nRanks 4 nNodes 2 localRanks 2 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] -1/-1/-1->1->0 [5] -1/-1/-1->1->0 [6] -1/-1/-1->1->0 [7] -1/-1/-1->1->0 [8] -1/-1/-1->1->0 [9] -1/-1/-1->1->0 [10] -1/-1/-1->1->0 [11] -1/-1/-1->1->0 [12] -1/-1/-1->1->0 [13] -1/-1/-1->1->0 [14] -1/-1/-1->1->0 [15] -1/-1/-1->1->0 [16] -1/-1/-1->1->0 [17] -1/-1/-1->1->0 [18] -1/-1/-1->1->0 [19] -1/-1/-1->1->0 [20] -1/-1/-1->1->0 [21] -1/-1/-1->1->0 [22] -1/-1/-1->1->0 [23] -1/-1/-1->1->0 [24] -1/-1/-1->1->0 [25] -1/-1/-1->1->0 [26] -1/-1/-1->1->0 [27] -1/-1/-1->1->0 [28] -1/-1/-1->1->0 [29] -1/-1/-1->1->0 [30] -1/-1/-1->1->0 [31] -1/-1/-1->1->0 comm 0x55ab4948db90 nRanks 04 busId f5000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Trees [0] 1/2/-1->0->-1 [1] 1/-1/-1->0->2 [2] 1/2/-1->0->-1 [3] 1/-1/-1->0->2 [4] 1/2/-1->0->-1 [5] 1/-1/-1->0->2 [6] 1/2/-1->0->-1 [7] 1/-1/-1->0->2 [8] 1/2/-1->0->-1 [9] 1/-1/-1->0->2 [10] 1/2/-1->0->-1 [11] 1/-1/-1->0->2 [12] 1/2/-1->0->-1 [13] 1/-1/-1->0->2 [14] 1/2/-1->0->-1 [15] 1/-1/-1->0->2 [16] 1/2/-1->0->-1 [17] 1/-1/-1->0->2 [18] 1/2/-1->0->-1 [19] 1/-1/-1->0->2 [20] 1/2/-1->0->-1 [21] 1/-1/-1->0->2 [22] 1/2/-1->0->-1 [23] 1/-1/-1->0->2 [24] 1/2/-1->0->-1 [25] 1/-1/-1->0->2 [26] 1/2/-1->0->-1 [27] 1/-1/-1->0->2 [28] 1/2/-1->0->-1 [29] 1/-1/-1->0->2 [30] 1/2/-1->0->-1 [31] 1/-1/-1->0->2 comm 0x5602714d6080 nRanks 04 busId 77000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Setting affinity for GPU 6 to ffff,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 00/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 01/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 02/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 03/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 04/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 05/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 06/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 07/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 08/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 09/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 10/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 11/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 12/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 13/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 14/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 15/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 16/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 17/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 18/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 19/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 20/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 21/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 22/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 23/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 24/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 25/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 26/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 27/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Setting affinity for GPU 5 to ffff0000,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 28/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 29/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 30/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 31/0 : 3[f5000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 00/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Setting affinity for GPU 2 to ffff,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Setting affinity for GPU 1 to ffff0000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO comm 0x563c0944bcb0 rank 1 nRanks 4 nNodes 2 localRanks 2 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO comm 0x5586eabac5b0 rank 0 nRanks 4 nNodes 2 localRanks 2 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] -1/-1/-1->1->0 [5] -1/-1/-1->1->0 [6] -1/-1/-1->1->0 [7] -1/-1/-1->1->0 [8] -1/-1/-1->1->0 [9] -1/-1/-1->1->0 [10] -1/-1/-1->1->0 [11] -1/-1/-1->1->0 [12] -1/-1/-1->1->0 [13] -1/-1/-1->1->0 [14] -1/-1/-1->1->0 [15] -1/-1/-1->1->0 [16] -1/-1/-1->1->0 [17] -1/-1/-1->1->0 [18] -1/-1/-1->1->0 [19] -1/-1/-1->1->0 [20] -1/-1/-1->1->0 [21] -1/-1/-1->1->0 [22] -1/-1/-1->1->0 [23] -1/-1/-1->1->0 [24] -1/-1/-1->1->0 [25] -1/-1/-1->1->0 [26] -1/-1/-1->1->0 [27] -1/-1/-1->1->0 [28] -1/-1/-1->1->0 [29] -1/-1/-1->1->0 [30] -1/-1/-1->1->0 [31] -1/-1/-1->1->0 comm 0x563c0944bcb0 nRanks 04 busId b5000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Trees [0] 1/2/-1->0->-1 [1] 1/-1/-1->0->2 [2] 1/2/-1->0->-1 [3] 1/-1/-1->0->2 [4] 1/2/-1->0->-1 [5] 1/-1/-1->0->2 [6] 1/2/-1->0->-1 [7] 1/-1/-1->0->2 [8] 1/2/-1->0->-1 [9] 1/-1/-1->0->2 [10] 1/2/-1->0->-1 [11] 1/-1/-1->0->2 [12] 1/2/-1->0->-1 [13] 1/-1/-1->0->2 [14] 1/2/-1->0->-1 [15] 1/-1/-1->0->2 [16] 1/2/-1->0->-1 [17] 1/-1/-1->0->2 [18] 1/2/-1->0->-1 [19] 1/-1/-1->0->2 [20] 1/2/-1->0->-1 [21] 1/-1/-1->0->2 [22] 1/2/-1->0->-1 [23] 1/-1/-1->0->2 [24] 1/2/-1->0->-1 [25] 1/-1/-1->0->2 [26] 1/2/-1->0->-1 [27] 1/-1/-1->0->2 [28] 1/2/-1->0->-1 [29] 1/-1/-1->0->2 [30] 1/2/-1->0->-1 [31] 1/-1/-1->0->2 comm 0x5586eabac5b0 nRanks 04 busId 36000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO comm 0x563da60624e0 rank 0 nRanks 4 nNodes 2 localRanks 2 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO comm 0x5612cbb74d90 rank 1 nRanks 4 nNodes 2 localRanks 2 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] -1/-1/-1->1->0 [5] -1/-1/-1->1->0 [6] -1/-1/-1->1->0 [7] -1/-1/-1->1->0 [8] -1/-1/-1->1->0 [9] -1/-1/-1->1->0 [10] -1/-1/-1->1->0 [11] -1/-1/-1->1->0 [12] -1/-1/-1->1->0 [13] -1/-1/-1->1->0 [14] -1/-1/-1->1->0 [15] -1/-1/-1->1->0 [16] -1/-1/-1->1->0 [17] -1/-1/-1->1->0 [18] -1/-1/-1->1->0 [19] -1/-1/-1->1->0 [20] -1/-1/-1->1->0 [21] -1/-1/-1->1->0 [22] -1/-1/-1->1->0 [23] -1/-1/-1->1->0 [24] -1/-1/-1->1->0 [25] -1/-1/-1->1->0 [26] -1/-1/-1->1->0 [27] -1/-1/-1->1->0 [28] -1/-1/-1->1->0 [29] -1/-1/-1->1->0 [30] -1/-1/-1->1->0 [31] -1/-1/-1->1->0 comm 0x5612cbb74d90 nRanks 04 busId d5000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Trees [0] 1/2/-1->0->-1 [1] 1/-1/-1->0->2 [2] 1/2/-1->0->-1 [3] 1/-1/-1->0->2 [4] 1/2/-1->0->-1 [5] 1/-1/-1->0->2 [6] 1/2/-1->0->-1 [7] 1/-1/-1->0->2 [8] 1/2/-1->0->-1 [9] 1/-1/-1->0->2 [10] 1/2/-1->0->-1 [11] 1/-1/-1->0->2 [12] 1/2/-1->0->-1 [13] 1/-1/-1->0->2 [14] 1/2/-1->0->-1 [15] 1/-1/-1->0->2 [16] 1/2/-1->0->-1 [17] 1/-1/-1->0->2 [18] 1/2/-1->0->-1 [19] 1/-1/-1->0->2 [20] 1/2/-1->0->-1 [21] 1/-1/-1->0->2 [22] 1/2/-1->0->-1 [23] 1/-1/-1->0->2 [24] 1/2/-1->0->-1 [25] 1/-1/-1->0->2 [26] 1/2/-1->0->-1 [27] 1/-1/-1->0->2 [28] 1/2/-1->0->-1 [29] 1/-1/-1->0->2 [30] 1/2/-1->0->-1 [31] 1/-1/-1->0->2 comm 0x563da60624e0 nRanks 04 busId 55000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Setting affinity for GPU 0 to ffff
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Setting affinity for GPU 4 to ffff,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 01/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 00/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 01/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 02/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 02/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 03/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 04/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 05/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 06/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 07/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 03/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 08/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 09/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 10/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 11/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 12/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 13/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 04/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO comm 0x556943960410 rank 0 nRanks 4 nNodes 2 localRanks 2 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 14/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO comm 0x5631932ed400 rank 1 nRanks 4 nNodes 2 localRanks 2 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 15/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] -1/-1/-1->1->0 [5] -1/-1/-1->1->0 [6] -1/-1/-1->1->0 [7] -1/-1/-1->1->0 [8] -1/-1/-1->1->0 [9] -1/-1/-1->1->0 [10] -1/-1/-1->1->0 [11] -1/-1/-1->1->0 [12] -1/-1/-1->1->0 [13] -1/-1/-1->1->0 [14] -1/-1/-1->1->0 [15] -1/-1/-1->1->0 [16] -1/-1/-1->1->0 [17] -1/-1/-1->1->0 [18] -1/-1/-1->1->0 [19] -1/-1/-1->1->0 [20] -1/-1/-1->1->0 [21] -1/-1/-1->1->0 [22] -1/-1/-1->1->0 [23] -1/-1/-1->1->0 [24] -1/-1/-1->1->0 [25] -1/-1/-1->1->0 [26] -1/-1/-1->1->0 [27] -1/-1/-1->1->0 [28] -1/-1/-1->1->0 [29] -1/-1/-1->1->0 [30] -1/-1/-1->1->0 [31] -1/-1/-1->1->0 comm 0x5631932ed400 nRanks 04 busId 85000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Trees [0] 1/2/-1->0->-1 [1] 1/-1/-1->0->2 [2] 1/2/-1->0->-1 [3] 1/-1/-1->0->2 [4] 1/2/-1->0->-1 [5] 1/-1/-1->0->2 [6] 1/2/-1->0->-1 [7] 1/-1/-1->0->2 [8] 1/2/-1->0->-1 [9] 1/-1/-1->0->2 [10] 1/2/-1->0->-1 [11] 1/-1/-1->0->2 [12] 1/2/-1->0->-1 [13] 1/-1/-1->0->2 [14] 1/2/-1->0->-1 [15] 1/-1/-1->0->2 [16] 1/2/-1->0->-1 [17] 1/-1/-1->0->2 [18] 1/2/-1->0->-1 [19] 1/-1/-1->0->2 [20] 1/2/-1->0->-1 [21] 1/-1/-1->0->2 [22] 1/2/-1->0->-1 [23] 1/-1/-1->0->2 [24] 1/2/-1->0->-1 [25] 1/-1/-1->0->2 [26] 1/2/-1->0->-1 [27] 1/-1/-1->0->2 [28] 1/2/-1->0->-1 [29] 1/-1/-1->0->2 [30] 1/2/-1->0->-1 [31] 1/-1/-1->0->2 comm 0x556943960410 nRanks 04 busId 9000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 16/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 17/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 18/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 19/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 05/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 20/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 21/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 22/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 23/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 24/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 25/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 06/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 26/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 27/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 28/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 29/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 30/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 31/0 : 3[d5000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 00/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 07/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 08/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 09/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 10/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 11/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 12/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 13/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 00/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 01/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 14/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 02/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 03/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 04/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 05/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 06/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 15/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 07/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 08/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 09/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 10/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 11/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 12/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 16/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 17/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 18/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 19/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 00/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 01/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 02/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 03/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 04/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 20/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 05/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 06/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 07/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 08/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 09/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 21/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 10/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 11/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 12/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 13/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 14/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 15/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 22/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 16/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 17/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 18/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 19/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 20/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 21/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 23/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 22/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 23/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 24/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 25/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 26/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 27/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 24/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 28/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 29/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 30/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 31/0 : 3[85000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 00/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 25/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 26/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 27/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 28/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 29/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 00/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 01/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 02/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 03/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 04/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 05/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 06/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 07/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 30/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 08/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 09/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 10/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 11/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 12/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 13/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 14/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 15/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 31/0 : 0[77000] -> 1[f5000] via P2P/IPC comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 16/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 17/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 18/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 19/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 20/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 21/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 22/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 23/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 24/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 25/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 26/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 27/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 28/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 29/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 30/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 31/0 : 1[f5000] -> 2[77000] [send] via NET/Socket/0 comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 01/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 02/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 03/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 04/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 05/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 06/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 07/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 08/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 09/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 10/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 11/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 12/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 13/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 14/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 15/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 01/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 16/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 00/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 01/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 02/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 02/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 03/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 04/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 05/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 17/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 06/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 07/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 08/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 09/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 10/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 11/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 03/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 12/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 13/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 18/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 14/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 15/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 16/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 17/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 18/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 19/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 04/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 20/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 21/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 22/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 19/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 23/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 24/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 25/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 26/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 27/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 28/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 05/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 29/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 30/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 20/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 31/0 : 1[d5000] -> 2[55000] [send] via NET/Socket/0 comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 06/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 21/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 07/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 22/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 00/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 01/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 02/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 03/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 08/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 04/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 05/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 06/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 23/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 07/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 08/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 09/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 10/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 11/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 12/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 09/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 13/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 14/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 24/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 15/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 16/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 17/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 18/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 19/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 20/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 21/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 10/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 22/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 23/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 25/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 24/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 25/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 26/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 27/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 28/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 29/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 11/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 30/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 26/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 31/0 : 1[b5000] -> 2[36000] [send] via NET/Socket/0 comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 12/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 27/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 28/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 13/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 29/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 14/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 30/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 15/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 31/0 : 0[55000] -> 1[d5000] via P2P/IPC comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 16/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 17/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 18/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 19/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 20/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 21/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 22/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 23/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 24/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 25/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 26/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 27/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 28/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 29/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 30/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 31/0 : 0[9000] -> 1[85000] via P2P/IPC comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 00/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 01/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 02/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 03/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 04/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 05/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 06/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 07/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 08/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 09/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 10/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 11/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 12/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 13/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 14/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 15/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 16/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 17/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 18/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 19/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 20/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 21/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 22/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 23/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 24/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 25/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 26/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 27/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 28/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 29/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 30/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 31/0 : 1[85000] -> 2[9000] [send] via NET/Socket/0 comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 00/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 01/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 02/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 03/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 04/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 05/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 06/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 07/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 00/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 08/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 09/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 10/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 11/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 12/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 13/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 14/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 13/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 14/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 15/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 16/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 17/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 18/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 19/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 20/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 21/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 22/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 23/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 24/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 25/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 26/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 27/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 28/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 29/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 30/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 31/0 : 3[b5000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 00/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 00/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 01/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 02/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 03/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 04/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 05/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 00/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 06/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 07/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 08/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 09/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 10/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 11/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 12/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 13/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 14/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 15/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 16/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 17/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 18/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 19/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 20/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 21/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 22/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 23/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 24/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 25/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 26/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 27/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 28/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 29/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 30/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 31/0 : 2[77000] -> 0[77000] [receive] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 00/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 01/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 02/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 03/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 01/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 04/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 05/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 06/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 07/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 08/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 09/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 10/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 11/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 12/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 13/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 02/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 14/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 15/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 16/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 17/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 18/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 19/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 20/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 21/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 22/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 03/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 23/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 24/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 25/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 26/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 27/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 28/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 29/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 30/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Channel 31/0 : 0[77000] -> 2[77000] [send] via NET/Socket/0 comm 0x5602714d6080 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 04/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 05/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 01/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 06/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 02/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 01/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 07/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 03/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 08/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 02/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 04/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 03/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 09/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 05/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 04/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 10/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 06/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 05/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 11/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 07/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 06/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 12/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 08/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 07/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 13/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 09/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 08/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 14/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 10/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 09/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 15/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 11/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 10/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 16/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 12/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 11/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 17/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 13/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 12/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 18/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 14/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 13/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 19/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 14/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 15/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 15/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 16/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 20/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 16/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 17/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 21/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 17/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 18/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 22/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 18/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 19/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 23/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 24/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 25/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 26/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 27/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 28/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 29/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 30/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 20/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Channel 31/0 : 1[f5000] -> 0[77000] via P2P/IPC comm 0x55ab4948db90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 21/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 22/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 23/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 24/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 19/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 25/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 20/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 26/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 21/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 27/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 22/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 28/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 23/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 29/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 24/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 30/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 25/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Channel 31/0 : 1[d5000] -> 0[55000] via P2P/IPC comm 0x5612cbb74d90 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 26/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 27/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 15/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 16/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 17/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 18/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 19/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 28/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 20/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 21/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 22/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 23/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 24/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 29/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 25/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 26/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 27/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 28/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 29/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 30/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 30/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 31/0 : 2[55000] -> 0[55000] [receive] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 00/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 01/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 02/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 03/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 04/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 05/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 31/0 : 0[36000] -> 1[b5000] via P2P/IPC comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 06/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 07/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 08/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 09/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 10/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 11/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 12/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 13/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 14/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 15/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 16/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 17/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 18/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 19/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 20/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 21/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 22/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 23/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 24/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 25/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 26/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 27/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 28/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 29/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 30/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Channel 31/0 : 0[55000] -> 2[55000] [send] via NET/Socket/0 comm 0x563da60624e0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 00/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 00/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 01/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 02/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 03/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 04/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 05/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 06/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 07/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 08/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 09/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 10/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 11/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 12/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 13/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 14/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 01/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 02/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 03/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 04/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 05/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 06/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 07/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 08/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 09/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 10/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 11/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 12/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 13/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 14/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 15/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 16/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 17/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 18/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 19/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 20/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 21/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 22/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 23/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 24/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 25/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 15/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 16/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 26/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 17/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 18/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 19/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 20/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 21/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 22/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 23/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 27/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 24/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 25/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 26/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 27/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 28/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 29/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 28/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 30/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 31/0 : 2[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 00/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 01/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 02/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 03/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 04/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 05/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 06/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 29/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 07/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 08/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 09/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 10/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 11/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 12/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 13/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 14/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 15/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 16/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 30/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 17/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 18/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 19/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 20/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 21/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 22/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 23/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 24/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 25/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 26/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Channel 31/0 : 1[85000] -> 0[9000] via P2P/IPC comm 0x5631932ed400 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 27/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 28/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 29/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 30/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Channel 31/0 : 0[9000] -> 2[9000] [send] via NET/Socket/0 comm 0x556943960410 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 00/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 00/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 01/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 02/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 03/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 04/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 05/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 06/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 07/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 08/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 09/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 10/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 11/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 12/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 13/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 14/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 15/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 16/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 17/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 18/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 19/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 20/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 21/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 22/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 23/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 24/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 25/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 26/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 27/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 28/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 29/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 30/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 31/0 : 2[36000] -> 0[36000] [receive] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 00/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 01/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 02/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 03/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 04/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 05/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 06/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 07/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 08/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 09/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 10/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 11/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 12/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 13/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 14/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 15/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 16/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 17/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 18/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 19/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 20/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 21/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 22/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 23/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 24/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 25/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 26/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 27/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 28/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 29/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 30/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Channel 31/0 : 0[36000] -> 2[36000] [send] via NET/Socket/0 comm 0x5586eabac5b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 01/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 02/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 03/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 04/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 05/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 06/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 07/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 08/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 09/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 10/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 11/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 12/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 13/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 14/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 15/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 16/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 17/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 18/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 19/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 20/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 21/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 22/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 23/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 24/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 25/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 26/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 27/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 28/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 29/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 30/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Channel 31/0 : 1[b5000] -> 0[36000] via P2P/IPC comm 0x563c0944bcb0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO ncclCommInitRank comm 0x5602714d6080 rank 0 nranks 4 cudaDev 3 nvmlDev 3 busId 77000 commId 0x47c875b4b070469a localSize 480 used 691521104 bytes on core 16 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7312 [3] NCCL INFO Init timings: rank 0 nranks 4 total 0.93 (kernels 0.00, bootstrap 0.40, allgathers 0.00, topo 0.22, graphs 0.00, connections 0.31, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO ncclCommInitRank comm 0x55ab4948db90 rank 1 nranks 4 cudaDev 7 nvmlDev 7 busId f5000 commId 0x47c875b4b070469a localSize 480 used 624412240 bytes on core 82 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7318 [7] NCCL INFO Init timings: rank 1 nranks 4 total 0.80 (kernels 0.00, bootstrap 0.27, allgathers 0.01, topo 0.21, graphs 0.00, connections 0.31, rest 0.00)
I1007 01:51:58.425478   353 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 7(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] NCCL_DEBUG: INFO
I1007 01:51:58.425601   357 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 7(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO ncclCommInitRank comm 0x5612cbb74d90 rank 1 nranks 4 cudaDev 6 nvmlDev 6 busId d5000 commId 0x5ecafbf2e3c52cd0 localSize 480 used 691521104 bytes on core 111 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7316 [6] NCCL INFO Init timings: rank 1 nranks 4 total 0.83 (kernels 0.00, bootstrap 0.27, allgathers 0.02, topo 0.22, graphs 0.00, connections 0.30, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO ncclCommInitRank comm 0x563da60624e0 rank 0 nranks 4 cudaDev 2 nvmlDev 2 busId 55000 commId 0x5ecafbf2e3c52cd0 localSize 480 used 691521104 bytes on core 45 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7314 [2] NCCL INFO Init timings: rank 0 nranks 4 total 0.94 (kernels 0.00, bootstrap 0.39, allgathers 0.00, topo 0.24, graphs 0.00, connections 0.30, rest 0.01)
I1007 01:51:58.441689   356 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 5(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] NCCL_DEBUG: INFO
I1007 01:51:58.441759   352 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 5(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO ncclCommInitRank comm 0x5631932ed400 rank 1 nranks 4 cudaDev 4 nvmlDev 4 busId 85000 commId 0x6f61729fbd02ce20 localSize 480 used 691521104 bytes on core 67 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7317 [4] NCCL INFO Init timings: rank 1 nranks 4 total 1.14 (kernels 0.00, bootstrap 0.30, allgathers 0.01, topo 0.23, graphs 0.00, connections 0.60, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO ncclCommInitRank comm 0x556943960410 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 9000 commId 0x6f61729fbd02ce20 localSize 480 used 634766928 bytes on core 7 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7313 [0] NCCL INFO Init timings: rank 0 nranks 4 total 1.26 (kernels 0.00, bootstrap 0.42, allgathers 0.01, topo 0.22, graphs 0.00, connections 0.60, rest 0.00)
I1007 01:51:58.755362   350 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] NCCL_DEBUG: INFO
I1007 01:51:58.755374   354 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 1(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO ncclCommInitRank comm 0x5586eabac5b0 rank 0 nranks 4 cudaDev 1 nvmlDev 1 busId 36000 commId 0x816cb0dbd678753f localSize 480 used 691521104 bytes on core 48 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7311 [1] NCCL INFO Init timings: rank 0 nranks 4 total 1.34 (kernels 0.00, bootstrap 0.42, allgathers 0.00, topo 0.22, graphs 0.00, connections 0.69, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO ncclCommInitRank comm 0x563c0944bcb0 rank 1 nranks 4 cudaDev 5 nvmlDev 5 busId b5000 commId 0x816cb0dbd678753f localSize 480 used 691521104 bytes on core 112 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7315 [5] NCCL INFO Init timings: rank 1 nranks 4 total 1.22 (kernels 0.00, bootstrap 0.31, allgathers 0.01, topo 0.22, graphs 0.00, connections 0.69, rest 0.00)
I1007 01:51:58.826371   351 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 3(DATA_PARALLEL_GROUP_WITH_CP) Rank 0] NCCL_DEBUG: INFO
I1007 01:51:58.826507   355 ProcessGroupNCCL.cpp:2890] [PG ID 1 PG GUID 3(DATA_PARALLEL_GROUP_WITH_CP) Rank 1] NCCL_DEBUG: INFO
I1007 01:55:07.325881   354 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 4] Using non-blocking mode: 0
I1007 01:55:07.326049   352 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 2] Using non-blocking mode: 0
I1007 01:55:07.326200   355 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 5] Using non-blocking mode: 0
I1007 01:55:07.326213   356 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 6] Using non-blocking mode: 0
I1007 01:55:07.326383   350 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:55:07.326556   353 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 3] Using non-blocking mode: 0
I1007 01:55:07.326712   357 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 7] Using non-blocking mode: 0
I1007 01:55:07.326776   350 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.035721 ms
I1007 01:55:07.326797   350 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:55:07.327013   351 ProcessGroupNCCL.cpp:1107] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 1] Using non-blocking mode: 0
I1007 01:55:07.327071   352 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 0.986129 ms
I1007 01:55:07.327101   352 NCCLUtils.cpp:75] Rank 2: creating NCCL communicator with mode: blocking
I1007 01:55:07.327082   354 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 4] ProcessGroupNCCL broadcast unique ID through store took 1.14747 ms
I1007 01:55:07.327148   354 NCCLUtils.cpp:75] Rank 4: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Kernel version: 5.15.0-25-generic
I1007 01:55:07.327169   355 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 5] ProcessGroupNCCL broadcast unique ID through store took 0.934448 ms
I1007 01:55:07.327196   355 NCCLUtils.cpp:75] Rank 5: creating NCCL communicator with mode: blocking
I1007 01:55:07.327180   356 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 6] ProcessGroupNCCL broadcast unique ID through store took 0.938658 ms
I1007 01:55:07.327220   356 NCCLUtils.cpp:75] Rank 6: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:55:07.327241   353 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 0.651256 ms
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:55:07.327273   353 NCCLUtils.cpp:75] Rank 3: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:55:07.327374   357 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 7] ProcessGroupNCCL broadcast unique ID through store took 0.636826 ms
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:55:07.327417   357 NCCLUtils.cpp:75] Rank 7: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
I1007 01:55:07.327481   351 ProcessGroupNCCL.cpp:2857] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 0.415164 ms
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
I1007 01:55:07.327523   351 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO ncclCommInitRank comm 0x5612cbc93b10 rank 6 nranks 16 cudaDev 6 nvmlDev 6 busId d5000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO ncclCommInitRank comm 0x563c0956ab90 rank 5 nranks 16 cudaDev 5 nvmlDev 5 busId b5000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO ncclCommInitRank comm 0x55ab495ab7d0 rank 7 nranks 16 cudaDev 7 nvmlDev 7 busId f5000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO rank:6 get match hsa dev domain:0x0 bdf:0xd500 rank busId:0x872448
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO ncclCommInitRank comm 0x5631934172f0 rank 4 nranks 16 cudaDev 4 nvmlDev 4 busId 85000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO rank:5 get match hsa dev domain:0x0 bdf:0xb500 rank busId:0x741376
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO rank:7 get match hsa dev domain:0x0 bdf:0xf500 rank busId:0x1003520
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO rank:4 get match hsa dev domain:0x0 bdf:0x8500 rank busId:0x544768
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO ncclCommInitRank comm 0x563da61891e0 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 55000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO ncclCommInitRank comm 0x5602715f9df0 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 77000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO ncclCommInitRank comm 0x5586eacd1420 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 36000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO rank:2 get match hsa dev domain:0x0 bdf:0x5500 rank busId:0x348160
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO rank:3 get match hsa dev domain:0x0 bdf:0x7700 rank busId:0x487424
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0x3600 rank busId:0x221184
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO ncclCommInitRank comm 0x556943a89750 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 9000 commId 0x48b23a3579159d54 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x900 rank busId:0x36864
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Setting affinity for GPU 4 to ffff,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Rank:4 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Transport init rank 4 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Setting affinity for GPU 6 to ffff,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Rank:6 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Transport init rank 6 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Setting affinity for GPU 5 to ffff0000,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Rank:5 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Transport init rank 5 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Setting affinity for GPU 3 to ffff0000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Rank:3 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Transport init rank 3 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Setting affinity for GPU 1 to ffff0000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Setting affinity for GPU 0 to ffff
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Setting affinity for GPU 7 to ffff0000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Rank:7 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Transport init rank 7 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Setting affinity for GPU 2 to ffff,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Rank:2 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Transport init rank 2 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO comm 0x5612cbc93b10 rank 6 nRanks 16 nNodes 2 localRanks 8 localRank 6 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO comm 0x563c0956ab90 rank 5 nRanks 16 nNodes 2 localRanks 8 localRank 5 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO comm 0x5631934172f0 rank 4 nRanks 16 nNodes 2 localRanks 8 localRank 4 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO comm 0x5602715f9df0 rank 3 nRanks 16 nNodes 2 localRanks 8 localRank 3 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO comm 0x563da61891e0 rank 2 nRanks 16 nNodes 2 localRanks 8 localRank 2 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO comm 0x55ab495ab7d0 rank 7 nRanks 16 nNodes 2 localRanks 8 localRank 7 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Trees [0] 6/-1/-1->5->4 [1] 6/-1/-1->5->4 [2] 6/-1/-1->5->4 [3] 6/-1/-1->5->4 [4] 6/-1/-1->5->4 [5] 6/-1/-1->5->4 [6] 6/-1/-1->5->4 [7] 6/-1/-1->5->4 [8] 6/-1/-1->5->4 [9] 6/-1/-1->5->4 [10] 6/-1/-1->5->4 [11] 6/-1/-1->5->4 [12] 6/-1/-1->5->4 [13] 6/-1/-1->5->4 [14] 6/-1/-1->5->4 [15] 6/-1/-1->5->4 [16] 6/-1/-1->5->4 [17] 6/-1/-1->5->4 [18] 6/-1/-1->5->4 [19] 6/-1/-1->5->4 [20] 6/-1/-1->5->4 [21] 6/-1/-1->5->4 [22] 6/-1/-1->5->4 [23] 6/-1/-1->5->4 [24] 6/-1/-1->5->4 [25] 6/-1/-1->5->4 [26] 6/-1/-1->5->4 [27] 6/-1/-1->5->4 [28] 6/-1/-1->5->4 [29] 6/-1/-1->5->4 [30] 6/-1/-1->5->4 [31] 6/-1/-1->5->4 comm 0x563c0956ab90 nRanks 16 busId b5000
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO comm 0x5586eacd1420 rank 1 nRanks 16 nNodes 2 localRanks 8 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Trees [0] 7/-1/-1->6->5 [1] 7/-1/-1->6->5 [2] 7/-1/-1->6->5 [3] 7/-1/-1->6->5 [4] 7/-1/-1->6->5 [5] 7/-1/-1->6->5 [6] 7/-1/-1->6->5 [7] 7/-1/-1->6->5 [8] 7/-1/-1->6->5 [9] 7/-1/-1->6->5 [10] 7/-1/-1->6->5 [11] 7/-1/-1->6->5 [12] 7/-1/-1->6->5 [13] 7/-1/-1->6->5 [14] 7/-1/-1->6->5 [15] 7/-1/-1->6->5 [16] 7/-1/-1->6->5 [17] 7/-1/-1->6->5 [18] 7/-1/-1->6->5 [19] 7/-1/-1->6->5 [20] 7/-1/-1->6->5 [21] 7/-1/-1->6->5 [22] 7/-1/-1->6->5 [23] 7/-1/-1->6->5 [24] 7/-1/-1->6->5 [25] 7/-1/-1->6->5 [26] 7/-1/-1->6->5 [27] 7/-1/-1->6->5 [28] 7/-1/-1->6->5 [29] 7/-1/-1->6->5 [30] 7/-1/-1->6->5 [31] 7/-1/-1->6->5 comm 0x5612cbc93b10 nRanks 16 busId d5000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO comm 0x556943a89750 rank 0 nRanks 16 nNodes 2 localRanks 8 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->1 [3] 3/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->1 [7] 3/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->1 [11] 3/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->1 [15] 3/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->1 [19] 3/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->1 [23] 3/-1/-1->2->1 [24] 3/-1/-1->2->1 [25] 3/-1/-1->2->1 [26] 3/-1/-1->2->1 [27] 3/-1/-1->2->1 [28] 3/-1/-1->2->1 [29] 3/-1/-1->2->1 [30] 3/-1/-1->2->1 [31] 3/-1/-1->2->1 comm 0x563da61891e0 nRanks 16 busId 55000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Trees [0] -1/-1/-1->7->6 [1] -1/-1/-1->7->6 [2] -1/-1/-1->7->6 [3] -1/-1/-1->7->6 [4] -1/-1/-1->7->6 [5] -1/-1/-1->7->6 [6] -1/-1/-1->7->6 [7] -1/-1/-1->7->6 [8] -1/-1/-1->7->6 [9] -1/-1/-1->7->6 [10] -1/-1/-1->7->6 [11] -1/-1/-1->7->6 [12] -1/-1/-1->7->6 [13] -1/-1/-1->7->6 [14] -1/-1/-1->7->6 [15] -1/-1/-1->7->6 [16] -1/-1/-1->7->6 [17] -1/-1/-1->7->6 [18] -1/-1/-1->7->6 [19] -1/-1/-1->7->6 [20] -1/-1/-1->7->6 [21] -1/-1/-1->7->6 [22] -1/-1/-1->7->6 [23] -1/-1/-1->7->6 [24] -1/-1/-1->7->6 [25] -1/-1/-1->7->6 [26] -1/-1/-1->7->6 [27] -1/-1/-1->7->6 [28] -1/-1/-1->7->6 [29] -1/-1/-1->7->6 [30] -1/-1/-1->7->6 [31] -1/-1/-1->7->6 comm 0x55ab495ab7d0 nRanks 16 busId f5000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Trees [0] 5/-1/-1->4->3 [1] 5/-1/-1->4->3 [2] 5/-1/-1->4->3 [3] 5/-1/-1->4->3 [4] 5/-1/-1->4->3 [5] 5/-1/-1->4->3 [6] 5/-1/-1->4->3 [7] 5/-1/-1->4->3 [8] 5/-1/-1->4->3 [9] 5/-1/-1->4->3 [10] 5/-1/-1->4->3 [11] 5/-1/-1->4->3 [12] 5/-1/-1->4->3 [13] 5/-1/-1->4->3 [14] 5/-1/-1->4->3 [15] 5/-1/-1->4->3 [16] 5/-1/-1->4->3 [17] 5/-1/-1->4->3 [18] 5/-1/-1->4->3 [19] 5/-1/-1->4->3 [20] 5/-1/-1->4->3 [21] 5/-1/-1->4->3 [22] 5/-1/-1->4->3 [23] 5/-1/-1->4->3 [24] 5/-1/-1->4->3 [25] 5/-1/-1->4->3 [26] 5/-1/-1->4->3 [27] 5/-1/-1->4->3 [28] 5/-1/-1->4->3 [29] 5/-1/-1->4->3 [30] 5/-1/-1->4->3 [31] 5/-1/-1->4->3 comm 0x5631934172f0 nRanks 16 busId 85000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Trees [0] 4/-1/-1->3->2 [1] 4/-1/-1->3->2 [2] 4/-1/-1->3->2 [3] 4/-1/-1->3->2 [4] 4/-1/-1->3->2 [5] 4/-1/-1->3->2 [6] 4/-1/-1->3->2 [7] 4/-1/-1->3->2 [8] 4/-1/-1->3->2 [9] 4/-1/-1->3->2 [10] 4/-1/-1->3->2 [11] 4/-1/-1->3->2 [12] 4/-1/-1->3->2 [13] 4/-1/-1->3->2 [14] 4/-1/-1->3->2 [15] 4/-1/-1->3->2 [16] 4/-1/-1->3->2 [17] 4/-1/-1->3->2 [18] 4/-1/-1->3->2 [19] 4/-1/-1->3->2 [20] 4/-1/-1->3->2 [21] 4/-1/-1->3->2 [22] 4/-1/-1->3->2 [23] 4/-1/-1->3->2 [24] 4/-1/-1->3->2 [25] 4/-1/-1->3->2 [26] 4/-1/-1->3->2 [27] 4/-1/-1->3->2 [28] 4/-1/-1->3->2 [29] 4/-1/-1->3->2 [30] 4/-1/-1->3->2 [31] 4/-1/-1->3->2 comm 0x5602715f9df0 nRanks 16 busId 77000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0 [2] 2/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->0 [6] 2/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->0 [10] 2/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->0 [14] 2/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->0 [18] 2/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->0 [22] 2/-1/-1->1->0 [23] 2/-1/-1->1->0 [24] 2/-1/-1->1->0 [25] 2/-1/-1->1->0 [26] 2/-1/-1->1->0 [27] 2/-1/-1->1->0 [28] 2/-1/-1->1->0 [29] 2/-1/-1->1->0 [30] 2/-1/-1->1->0 [31] 2/-1/-1->1->0 comm 0x5586eacd1420 nRanks 16 busId 36000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 00/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 01/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 02/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 03/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 04/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 05/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 06/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 07/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 08/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 09/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 10/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 11/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 12/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 13/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 14/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 15/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 16/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 17/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 18/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 19/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 20/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 21/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 22/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 23/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 24/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 25/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 26/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 27/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 28/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 29/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 30/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 31/32 :    0   7   6   5   4   3   2   1   8  15  14  13  12  11  10   9
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Trees [0] 1/8/-1->0->-1 [1] 1/-1/-1->0->8 [2] 1/8/-1->0->-1 [3] 1/-1/-1->0->8 [4] 1/8/-1->0->-1 [5] 1/-1/-1->0->8 [6] 1/8/-1->0->-1 [7] 1/-1/-1->0->8 [8] 1/8/-1->0->-1 [9] 1/-1/-1->0->8 [10] 1/8/-1->0->-1 [11] 1/-1/-1->0->8 [12] 1/8/-1->0->-1 [13] 1/-1/-1->0->8 [14] 1/8/-1->0->-1 [15] 1/-1/-1->0->8 [16] 1/8/-1->0->-1 [17] 1/-1/-1->0->8 [18] 1/8/-1->0->-1 [19] 1/-1/-1->0->8 [20] 1/8/-1->0->-1 [21] 1/-1/-1->0->8 [22] 1/8/-1->0->-1 [23] 1/-1/-1->0->8 [24] 1/8/-1->0->-1 [25] 1/-1/-1->0->8 [26] 1/8/-1->0->-1 [27] 1/-1/-1->0->8 [28] 1/8/-1->0->-1 [29] 1/-1/-1->0->8 [30] 1/8/-1->0->-1 [31] 1/-1/-1->0->8 comm 0x556943a89750 nRanks 16 busId 9000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO P2P Chunksize set to 131072
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 00/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 01/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 02/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 03/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 04/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 05/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 06/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 07/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 08/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 09/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 10/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 11/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 12/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 13/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 14/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 15/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 16/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 17/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 18/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 19/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 00/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 20/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 21/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 01/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 22/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 02/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 23/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 24/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 03/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 25/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 04/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 26/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 05/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 27/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 28/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 06/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 29/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 07/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 30/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 08/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 31/0 : 1[36000] -> 8[9000] [send] via NET/Socket/0 comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 09/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 10/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 11/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 12/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 13/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 14/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 15/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 16/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 17/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 18/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 19/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 20/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 21/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 22/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 23/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 24/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 25/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 26/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 27/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 28/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 29/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 30/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 31/0 : 9[36000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 00/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 01/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 02/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 03/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 04/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 05/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 06/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 07/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 08/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 09/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 10/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 11/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 00/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 12/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 01/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 13/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 02/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 14/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 03/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 00/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 15/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 04/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 01/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 16/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 05/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 02/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 17/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 06/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 03/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 18/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 00/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 07/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 04/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 19/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 01/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 08/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 05/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 20/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 02/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 09/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 06/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 03/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 21/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 10/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 07/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 00/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 04/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 22/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 11/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 08/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 01/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 05/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 12/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 09/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 23/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 02/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 00/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 10/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 06/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 13/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 24/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 03/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 01/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 11/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 14/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 07/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 25/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 04/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 12/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 02/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 15/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 08/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 26/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 05/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 13/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 03/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 16/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 09/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 27/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 06/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 14/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 17/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 04/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 10/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 28/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 15/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 07/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 18/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 05/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 11/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 29/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 16/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 08/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 19/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 06/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 12/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 30/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 17/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 09/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 20/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 07/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 13/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 18/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 31/0 : 0[9000] -> 7[f5000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 21/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 10/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 19/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 08/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 14/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 22/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 11/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 20/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 00/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 15/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 09/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 23/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 21/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 12/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 01/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 16/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 10/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 24/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 22/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 13/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 02/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 17/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 11/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 25/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 23/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 03/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 14/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 18/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 12/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 26/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 24/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 04/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 15/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 19/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 27/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 13/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 25/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 05/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 16/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 20/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 28/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 26/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 14/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 06/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 17/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 21/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 29/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 27/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 15/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 07/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 22/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 18/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 28/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 30/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 16/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 08/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 23/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 29/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 31/0 : 6[d5000] -> 5[b5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 19/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 17/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 09/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 30/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 24/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 20/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 10/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 31/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 25/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 21/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 18/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 11/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 26/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 22/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 19/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 27/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 20/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 23/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 12/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 28/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 24/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 21/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 13/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 29/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 25/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 14/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 22/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 30/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 15/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 26/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 23/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 31/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 16/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 27/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 24/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 17/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 28/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 25/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 18/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 26/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 29/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 19/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 27/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 30/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 20/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 28/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 31/0 : 5[b5000] -> 4[85000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 21/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 29/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 30/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 23/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 31/0 : 4[85000] -> 3[77000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 24/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 25/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 26/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 27/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 28/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 29/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 30/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Channel 31/0 : 7[f5000] -> 6[d5000] via P2P/IPC comm 0x55ab495ab7d0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 00/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 01/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 02/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 03/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 04/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 00/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 05/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 01/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 06/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 00/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 00/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 07/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 02/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 01/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 08/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 03/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 02/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 09/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 04/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 03/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 05/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 10/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 04/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 06/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 11/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 05/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 07/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 12/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 06/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 07/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 13/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 08/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 09/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 14/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 08/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 10/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 15/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 09/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 16/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 11/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 10/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 12/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 17/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 11/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 12/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 13/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 18/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 13/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 19/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 14/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 14/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 15/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 20/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 15/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 21/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 16/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 16/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 22/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 17/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 17/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 18/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 23/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 18/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 19/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 24/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 19/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 20/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 25/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 21/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 20/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 26/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 01/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 22/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 21/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 27/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 02/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 23/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 22/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 28/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 24/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 03/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 23/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 29/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 25/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 04/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 24/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 26/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 05/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 30/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 25/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 27/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 06/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Channel 31/0 : 4[85000] -> 5[b5000] via P2P/IPC comm 0x5631934172f0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 26/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 28/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 07/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 27/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 29/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 08/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 28/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 09/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 30/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 29/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 10/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Channel 31/0 : 3[77000] -> 4[85000] via P2P/IPC comm 0x5602715f9df0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 30/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 11/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Channel 31/0 : 5[b5000] -> 6[d5000] via P2P/IPC comm 0x563c0956ab90 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 12/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 13/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 14/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 15/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 16/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 17/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 18/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 19/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 20/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 21/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 22/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 23/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 24/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 25/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 26/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 27/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 28/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 29/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 30/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 31/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 00/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 01/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 02/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 03/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 04/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 05/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 06/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 07/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 08/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 09/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 10/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 00/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 11/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 01/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 12/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 02/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 13/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 00/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 03/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 14/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 04/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 01/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 15/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 05/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 02/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 16/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 06/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 03/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 17/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 07/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 04/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 08/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 18/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 05/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 09/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 19/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 06/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 10/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 20/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 07/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 11/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 21/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 12/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 08/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 22/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 13/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 09/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 23/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 14/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 10/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 24/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 15/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 11/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 25/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 16/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 12/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 26/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 17/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 13/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 18/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 27/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 19/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 14/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 28/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 20/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 15/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 29/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 21/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 16/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 30/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 22/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 17/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Channel 31/0 : 6[d5000] -> 7[f5000] via P2P/IPC comm 0x5612cbc93b10 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 23/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 18/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 24/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 19/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 25/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 20/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 26/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 21/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 27/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 22/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 28/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 23/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 29/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 24/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 30/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 25/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Channel 31/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da61891e0 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 26/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 27/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 28/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 29/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 30/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 31/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 00/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 00/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 01/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 02/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 03/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 04/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 05/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 06/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 01/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 07/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 08/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 09/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 10/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 11/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 12/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 13/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 02/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 14/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 15/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 16/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 17/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 18/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 19/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 20/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 03/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 21/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 22/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 04/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 05/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 06/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 07/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 08/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 23/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 24/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 25/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 26/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 27/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 28/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 09/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 29/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 30/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 31/0 : 8[9000] -> 0[9000] [receive] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 00/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 01/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 02/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 03/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 04/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 10/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 05/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 06/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 07/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 08/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 09/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 10/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 11/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 12/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 13/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 14/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 11/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 15/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 16/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 17/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 18/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 19/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 20/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 21/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 22/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 23/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 12/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 24/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 25/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 26/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 27/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 28/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 29/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 30/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Channel 31/0 : 0[9000] -> 8[9000] [send] via NET/Socket/0 comm 0x556943a89750 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 13/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 14/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 15/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 16/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 17/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 18/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 19/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 20/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 21/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 22/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 23/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 24/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 25/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 26/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 27/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 28/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 29/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 30/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Channel 31/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eacd1420 nRanks 16
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO threadThresholds 8/8/64 | 128/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 2 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO ncclCommInitRank comm 0x5586eacd1420 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 36000 commId 0x48b23a3579159d54 localSize 480 used 973899712 bytes on core 50 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO ncclCommInitRank comm 0x5602715f9df0 rank 3 nranks 16 cudaDev 3 nvmlDev 3 busId 77000 commId 0x48b23a3579159d54 localSize 480 used 973899712 bytes on core 30 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO ncclCommInitRank comm 0x5612cbc93b10 rank 6 nranks 16 cudaDev 6 nvmlDev 6 busId d5000 commId 0x48b23a3579159d54 localSize 480 used 973899712 bytes on core 101 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7348 [3] NCCL INFO Init timings: rank 3 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.06, topo 0.86, graphs 0.00, connections 0.25, rest 0.11)
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7350 [1] NCCL INFO Init timings: rank 1 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.05, topo 0.87, graphs 0.00, connections 0.36, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO ncclCommInitRank comm 0x563c0956ab90 rank 5 nranks 16 cudaDev 5 nvmlDev 5 busId b5000 commId 0x48b23a3579159d54 localSize 480 used 973899712 bytes on core 124 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7347 [6] NCCL INFO Init timings: rank 6 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.06, topo 0.86, graphs 0.00, connections 0.24, rest 0.12)
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7346 [5] NCCL INFO Init timings: rank 5 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.06, topo 0.86, graphs 0.00, connections 0.25, rest 0.11)
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO ncclCommInitRank comm 0x556943a89750 rank 0 nranks 16 cudaDev 0 nvmlDev 0 busId 9000 commId 0x48b23a3579159d54 localSize 480 used 860391360 bytes on core 0 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO ncclCommInitRank comm 0x55ab495ab7d0 rank 7 nranks 16 cudaDev 7 nvmlDev 7 busId f5000 commId 0x48b23a3579159d54 localSize 480 used 839681984 bytes on core 87 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO ncclCommInitRank comm 0x563da61891e0 rank 2 nranks 16 cudaDev 2 nvmlDev 2 busId 55000 commId 0x48b23a3579159d54 localSize 480 used 973899712 bytes on core 40 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO ncclCommInitRank comm 0x5631934172f0 rank 4 nranks 16 cudaDev 4 nvmlDev 4 busId 85000 commId 0x48b23a3579159d54 localSize 480 used 973899712 bytes on core 66 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7344 [0] NCCL INFO Init timings: rank 0 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.03, topo 0.89, graphs 0.00, connections 0.36, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7349 [7] NCCL INFO Init timings: rank 7 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.02, topo 0.89, graphs 0.00, connections 0.24, rest 0.12)
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7345 [2] NCCL INFO Init timings: rank 2 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.00, topo 0.91, graphs 0.00, connections 0.29, rest 0.07)
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7351 [4] NCCL INFO Init timings: rank 4 nranks 16 total 1.32 (kernels 0.00, bootstrap 0.04, allgathers 0.08, topo 0.83, graphs 0.00, connections 0.26, rest 0.11)
I1007 01:55:08.650895   356 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 6] NCCL_DEBUG: INFO
I1007 01:55:08.650944   351 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 1] NCCL_DEBUG: INFO
I1007 01:55:08.650952   350 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 0] NCCL_DEBUG: INFO
I1007 01:55:08.650962   352 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 2] NCCL_DEBUG: INFO
I1007 01:55:08.650981   353 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 3] NCCL_DEBUG: INFO
I1007 01:55:08.651005   355 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 5] NCCL_DEBUG: INFO
I1007 01:55:08.651091   354 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 4] NCCL_DEBUG: INFO
I1007 01:55:08.651129   357 ProcessGroupNCCL.cpp:2890] [PG ID 17 PG GUID 131(INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP) Rank 7] NCCL_DEBUG: INFO
I1007 01:56:41.273319   355 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 1] Using non-blocking mode: 0
I1007 01:56:41.290706   351 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 1] Using non-blocking mode: 0
I1007 01:56:41.422986   357 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 3] Using non-blocking mode: 0
I1007 01:56:41.432775   353 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 3] Using non-blocking mode: 0
I1007 01:56:41.486523   354 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:56:41.487031   354 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.046691 ms
I1007 01:56:41.487058   354 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:56:41.487185   357 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 64.1601 ms
I1007 01:56:41.487172   355 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 213.805 ms
I1007 01:56:41.487229   357 NCCLUtils.cpp:75] Rank 3: creating NCCL communicator with mode: blocking
I1007 01:56:41.487231   355 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Kernel version: 5.15.0-25-generic
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:354:354 [4] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:357:357 [7] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:355:355 [5] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:56:41.490401   350 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 0] Using non-blocking mode: 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:56:41.490770   350 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 0] ProcessGroupNCCL broadcast unique ID through store took 0.038471 ms
I1007 01:56:41.490790   350 NCCLUtils.cpp:75] Rank 0: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
I1007 01:56:41.490863   351 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 1] ProcessGroupNCCL broadcast unique ID through store took 200.107 ms
I1007 01:56:41.490929   351 NCCLUtils.cpp:75] Rank 1: creating NCCL communicator with mode: blocking
I1007 01:56:41.490932   353 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 3] ProcessGroupNCCL broadcast unique ID through store took 58.1217 ms
I1007 01:56:41.490980   353 NCCLUtils.cpp:75] Rank 3: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:350:350 [0] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:353 [3] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:351:351 [1] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:56:41.498601   356 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 2] Using non-blocking mode: 0
I1007 01:56:41.498854   356 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 0.211682 ms
I1007 01:56:41.498903   356 NCCLUtils.cpp:75] Rank 2: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:356:356 [6] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
I1007 01:56:41.500629   352 ProcessGroupNCCL.cpp:1107] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 2] Using non-blocking mode: 0
I1007 01:56:41.500797   352 ProcessGroupNCCL.cpp:2857] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 2] ProcessGroupNCCL broadcast unique ID through store took 0.138751 ms
I1007 01:56:41.500823   352 NCCLUtils.cpp:75] Rank 2: creating NCCL communicator with mode: blocking
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:170 NCCL WARN NUMA auto balancing enabled which can lead to variability in the RCCL performance! Disable by "sudo sysctl kernel.numa_balancing=0"
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Kernel version: 5.15.0-25-generic
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:191 NCCL WARN Missing "iommu=pt" from kernel command line which can lead to system instablity or hang!
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/init.cc:196 NCCL WARN Missing "HSA_FORCE_FINE_GRAIN_PCIE=1" from environment which can lead to low RCCL performance, system instablity or hang!
vcjob-2107528843008176130-abc12335-66689-worker-0:352:352 [2] NCCL INFO Comm config Blocking set to 1
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Using network Socket
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO ncclCommInitRank comm 0x563c09705ef0 rank 1 nranks 4 cudaDev 5 nvmlDev 5 busId b5000 commId 0x5152a155dda8145a - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO ncclCommInitRank comm 0x55ab49747fc0 rank 3 nranks 4 cudaDev 7 nvmlDev 7 busId f5000 commId 0x5152a155dda8145a - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO ncclCommInitRank comm 0x5631935b6f70 rank 0 nranks 4 cudaDev 4 nvmlDev 4 busId 85000 commId 0x5152a155dda8145a - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO rank:3 get match hsa dev domain:0x0 bdf:0xf500 rank busId:0x1003520
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x8500 rank busId:0x544768
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO ncclCommInitRank comm 0x5612cbe2ebf0 rank 2 nranks 4 cudaDev 6 nvmlDev 6 busId d5000 commId 0x5152a155dda8145a - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0xb500 rank busId:0x741376
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO rank:2 get match hsa dev domain:0x0 bdf:0xd500 rank busId:0x872448
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO ncclCommInitRank comm 0x563da6326960 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 55000 commId 0x74c3b44357f274b7 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO ncclCommInitRank comm 0x5602717929b0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 77000 commId 0x74c3b44357f274b7 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO ncclCommInitRank comm 0x5586eae69710 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 36000 commId 0x74c3b44357f274b7 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO ncclCommInitRank comm 0x556943c272b0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 9000 commId 0x74c3b44357f274b7 - Init START
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO rank:2 get match hsa dev domain:0x0 bdf:0x5500 rank busId:0x348160
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO rank:3 get match hsa dev domain:0x0 bdf:0x7700 rank busId:0x487424
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO rank:0 get match hsa dev domain:0x0 bdf:0x900 rank busId:0x36864
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO rank:1 get match hsa dev domain:0x0 bdf:0x3600 rank busId:0x221184
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO XDP Switch is off with xdp_size : 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO XDP Switch is off with xdp_size : 0
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] /data/jenkins_workspace/workspace/rccl_release/build/hipify/src/graph/xml.cc:1211 NCCL WARN No IB NIC found, the generation of the basic topology has been terminated.
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO rocm_smi_lib: version 7.6.0.0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Setting affinity for GPU 6 to ffff,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Rank:2 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Transport init rank 2 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Setting affinity for GPU 7 to ffff0000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Rank:3 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Transport init rank 3 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Setting affinity for GPU 5 to ffff0000,00000000,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Setting affinity for GPU 4 to ffff,00000000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO comm 0x5631935b6f70 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO comm 0x563c09705ef0 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO comm 0x55ab49747fc0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO comm 0x5612cbe2ebf0 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->-1 [2] -1/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->-1 [6] -1/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->-1 [10] -1/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->-1 [14] -1/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->-1 [18] -1/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->-1 [22] -1/-1/-1->1->0 [23] 2/-1/-1->1->0 [24] 2/-1/-1->1->0 [25] 2/-1/-1->1->-1 [26] -1/-1/-1->1->0 [27] 2/-1/-1->1->0 [28] 2/-1/-1->1->0 [29] 2/-1/-1->1->-1 [30] -1/-1/-1->1->0 [31] 2/-1/-1->1->0 comm 0x563c09705ef0 nRanks 04 busId b5000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] 0/-1/-1->3->2 [2] 0/-1/-1->3->2 [3] 0/-1/-1->3->-1 [4] -1/-1/-1->3->2 [5] 0/-1/-1->3->2 [6] 0/-1/-1->3->2 [7] 0/-1/-1->3->-1 [8] -1/-1/-1->3->2 [9] 0/-1/-1->3->2 [10] 0/-1/-1->3->2 [11] 0/-1/-1->3->-1 [12] -1/-1/-1->3->2 [13] 0/-1/-1->3->2 [14] 0/-1/-1->3->2 [15] 0/-1/-1->3->-1 [16] -1/-1/-1->3->2 [17] 0/-1/-1->3->2 [18] 0/-1/-1->3->2 [19] 0/-1/-1->3->-1 [20] -1/-1/-1->3->2 [21] 0/-1/-1->3->2 [22] 0/-1/-1->3->2 [23] 0/-1/-1->3->-1 [24] -1/-1/-1->3->2 [25] 0/-1/-1->3->2 [26] 0/-1/-1->3->2 [27] 0/-1/-1->3->-1 [28] -1/-1/-1->3->2 [29] 0/-1/-1->3->2 [30] 0/-1/-1->3->2 [31] 0/-1/-1->3->-1 comm 0x55ab49747fc0 nRanks 04 busId f5000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->-1 [3] -1/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->-1 [7] -1/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->-1 [11] -1/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->-1 [15] -1/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->-1 [19] -1/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->-1 [23] -1/-1/-1->2->1 [24] 3/-1/-1->2->1 [25] 3/-1/-1->2->1 [26] 3/-1/-1->2->-1 [27] -1/-1/-1->2->1 [28] 3/-1/-1->2->1 [29] 3/-1/-1->2->1 [30] 3/-1/-1->2->-1 [31] -1/-1/-1->2->1 comm 0x5612cbe2ebf0 nRanks 04 busId d5000
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->3 [2] 1/-1/-1->0->3 [3] 1/-1/-1->0->3 [4] 1/-1/-1->0->-1 [5] -1/-1/-1->0->3 [6] 1/-1/-1->0->3 [7] 1/-1/-1->0->3 [8] 1/-1/-1->0->-1 [9] -1/-1/-1->0->3 [10] 1/-1/-1->0->3 [11] 1/-1/-1->0->3 [12] 1/-1/-1->0->-1 [13] -1/-1/-1->0->3 [14] 1/-1/-1->0->3 [15] 1/-1/-1->0->3 [16] 1/-1/-1->0->-1 [17] -1/-1/-1->0->3 [18] 1/-1/-1->0->3 [19] 1/-1/-1->0->3 [20] 1/-1/-1->0->-1 [21] -1/-1/-1->0->3 [22] 1/-1/-1->0->3 [23] 1/-1/-1->0->3 [24] 1/-1/-1->0->-1 [25] -1/-1/-1->0->3 [26] 1/-1/-1->0->3 [27] 1/-1/-1->0->3 [28] 1/-1/-1->0->-1 [29] -1/-1/-1->0->3 [30] 1/-1/-1->0->3 [31] 1/-1/-1->0->3 comm 0x5631935b6f70 nRanks 04 busId 85000
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Setting affinity for GPU 3 to ffff0000
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Rank:3 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Transport init rank 3 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Setting affinity for GPU 2 to ffff,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Setting affinity for GPU 0 to ffff
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Rank:2 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Transport init rank 2 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Rank:0 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Transport init rank 0 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 00/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 01/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 02/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 03/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 04/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Setting affinity for GPU 1 to ffff0000,00000000
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Rank:1 Get link topo from rocmDev:0 linkNum:8 link:0 linkState:1 remote type:1 bdf:0x9b00 topoType:2
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Transport init rank 1 useXdp 0 linkTopoType 2
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 00/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 05/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 00/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 00/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 01/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 06/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO comm 0x556943c272b0 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO comm 0x563da6326960 rank 2 nRanks 4 nNodes 1 localRanks 4 localRank 2 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO comm 0x5586eae69710 rank 1 nRanks 4 nNodes 1 localRanks 4 localRank 1 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO comm 0x5602717929b0 rank 3 nRanks 4 nNodes 1 localRanks 4 localRank 3 MNNVL 0
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 00/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 01/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 02/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 03/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 04/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 05/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 06/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 07/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1 [2] 3/-1/-1->2->-1 [3] -1/-1/-1->2->1 [4] 3/-1/-1->2->1 [5] 3/-1/-1->2->1 [6] 3/-1/-1->2->-1 [7] -1/-1/-1->2->1 [8] 3/-1/-1->2->1 [9] 3/-1/-1->2->1 [10] 3/-1/-1->2->-1 [11] -1/-1/-1->2->1 [12] 3/-1/-1->2->1 [13] 3/-1/-1->2->1 [14] 3/-1/-1->2->-1 [15] -1/-1/-1->2->1 [16] 3/-1/-1->2->1 [17] 3/-1/-1->2->1 [18] 3/-1/-1->2->-1 [19] -1/-1/-1->2->1 [20] 3/-1/-1->2->1 [21] 3/-1/-1->2->1 [22] 3/-1/-1->2->-1 [23] -1/-1/-1->2->1 [24] 3/-1/-1->2->1 [25] 3/-1/-1->2->1 [26] 3/-1/-1->2->-1 [27] -1/-1/-1->2->1 [28] 3/-1/-1->2->1 [29] 3/-1/-1->2->1 [30] 3/-1/-1->2->-1 [31] -1/-1/-1->2->1 comm 0x563da6326960 nRanks 04 busId 55000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 08/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->-1 [2] -1/-1/-1->1->0 [3] 2/-1/-1->1->0 [4] 2/-1/-1->1->0 [5] 2/-1/-1->1->-1 [6] -1/-1/-1->1->0 [7] 2/-1/-1->1->0 [8] 2/-1/-1->1->0 [9] 2/-1/-1->1->-1 [10] -1/-1/-1->1->0 [11] 2/-1/-1->1->0 [12] 2/-1/-1->1->0 [13] 2/-1/-1->1->-1 [14] -1/-1/-1->1->0 [15] 2/-1/-1->1->0 [16] 2/-1/-1->1->0 [17] 2/-1/-1->1->-1 [18] -1/-1/-1->1->0 [19] 2/-1/-1->1->0 [20] 2/-1/-1->1->0 [21] 2/-1/-1->1->-1 [22] -1/-1/-1->1->0 [23] 2/-1/-1->1->0 [24] 2/-1/-1->1->0 [25] 2/-1/-1->1->-1 [26] -1/-1/-1->1->0 [27] 2/-1/-1->1->0 [28] 2/-1/-1->1->0 [29] 2/-1/-1->1->-1 [30] -1/-1/-1->1->0 [31] 2/-1/-1->1->0 comm 0x5586eae69710 nRanks 04 busId 36000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 09/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] 0/-1/-1->3->2 [2] 0/-1/-1->3->2 [3] 0/-1/-1->3->-1 [4] -1/-1/-1->3->2 [5] 0/-1/-1->3->2 [6] 0/-1/-1->3->2 [7] 0/-1/-1->3->-1 [8] -1/-1/-1->3->2 [9] 0/-1/-1->3->2 [10] 0/-1/-1->3->2 [11] 0/-1/-1->3->-1 [12] -1/-1/-1->3->2 [13] 0/-1/-1->3->2 [14] 0/-1/-1->3->2 [15] 0/-1/-1->3->-1 [16] -1/-1/-1->3->2 [17] 0/-1/-1->3->2 [18] 0/-1/-1->3->2 [19] 0/-1/-1->3->-1 [20] -1/-1/-1->3->2 [21] 0/-1/-1->3->2 [22] 0/-1/-1->3->2 [23] 0/-1/-1->3->-1 [24] -1/-1/-1->3->2 [25] 0/-1/-1->3->2 [26] 0/-1/-1->3->2 [27] 0/-1/-1->3->-1 [28] -1/-1/-1->3->2 [29] 0/-1/-1->3->2 [30] 0/-1/-1->3->2 [31] 0/-1/-1->3->-1 comm 0x5602717929b0 nRanks 04 busId 77000
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 10/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 11/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 12/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 13/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 14/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 15/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 16/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 02/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 17/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 18/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 19/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 20/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 21/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 22/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 01/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 23/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 24/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 25/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 01/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 26/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 27/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 28/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 07/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 29/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 30/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 31/32 :    0   1   2   3
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] -1/-1/-1->0->3 [2] 1/-1/-1->0->3 [3] 1/-1/-1->0->3 [4] 1/-1/-1->0->-1 [5] -1/-1/-1->0->3 [6] 1/-1/-1->0->3 [7] 1/-1/-1->0->3 [8] 1/-1/-1->0->-1 [9] -1/-1/-1->0->3 [10] 1/-1/-1->0->3 [11] 1/-1/-1->0->3 [12] 1/-1/-1->0->-1 [13] -1/-1/-1->0->3 [14] 1/-1/-1->0->3 [15] 1/-1/-1->0->3 [16] 1/-1/-1->0->-1 [17] -1/-1/-1->0->3 [18] 1/-1/-1->0->3 [19] 1/-1/-1->0->3 [20] 1/-1/-1->0->-1 [21] -1/-1/-1->0->3 [22] 1/-1/-1->0->3 [23] 1/-1/-1->0->3 [24] 1/-1/-1->0->-1 [25] -1/-1/-1->0->3 [26] 1/-1/-1->0->3 [27] 1/-1/-1->0->3 [28] 1/-1/-1->0->-1 [29] -1/-1/-1->0->3 [30] 1/-1/-1->0->3 [31] 1/-1/-1->0->3 comm 0x556943c272b0 nRanks 04 busId 9000
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO P2P Chunksize set to 262144
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO comm->autoPerPeerOn: 1, comm->p2pnChannels: 32, comm->p2pnChannelsPerPeer: 16
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 03/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 08/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 02/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 02/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 04/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 09/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 03/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 03/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 05/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 10/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 04/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 04/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 06/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 11/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 05/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 05/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 07/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 12/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 06/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 06/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 08/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 13/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 07/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 09/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 07/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 14/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 08/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 10/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 08/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 15/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 09/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 11/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 16/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 09/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 10/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 12/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 17/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 10/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 13/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 11/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 18/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 11/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 14/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 12/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 19/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 12/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 15/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 13/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 20/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 16/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 13/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 21/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 14/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 17/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 22/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 15/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 18/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 14/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 23/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 16/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 19/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 15/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 24/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 17/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 20/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 25/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 16/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 18/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 21/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 26/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 19/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 22/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 17/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 27/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 20/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 23/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 18/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 28/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 21/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 24/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 29/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 19/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 22/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 25/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 30/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 20/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 23/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 26/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 31/0 : 2[d5000] -> 3[f5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 21/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 24/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 27/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 22/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 28/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 25/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 29/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 23/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 26/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 30/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 27/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 24/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 31/0 : 3[f5000] -> 0[85000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 28/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 25/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 29/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 26/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 30/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 27/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 31/0 : 0[85000] -> 1[b5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 28/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 29/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 30/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 31/0 : 1[b5000] -> 2[d5000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 00/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 01/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 02/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 00/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 03/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 01/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 00/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 04/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 02/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 05/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 01/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 03/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 06/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 02/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 04/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 07/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 03/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 05/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 00/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 08/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 04/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 06/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 01/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 09/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 05/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 07/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 02/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 10/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 06/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 08/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 03/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 11/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 07/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 09/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 04/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 12/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 10/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 08/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 05/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 13/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 11/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 09/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 14/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 06/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 12/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 10/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 15/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 07/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 13/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 11/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 16/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 08/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 14/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 12/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 17/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 09/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 15/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 13/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 18/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 10/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 16/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 14/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 19/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 11/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 17/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 15/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 20/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 12/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 18/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 16/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 21/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 13/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 19/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 17/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 22/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 14/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 20/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 23/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 18/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 15/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 21/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 24/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 19/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 16/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 22/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 25/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 20/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 17/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 23/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 26/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 21/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 24/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 18/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 27/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 22/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 25/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 19/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 28/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 23/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 26/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 20/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 29/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 24/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 27/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 21/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 30/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 25/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 28/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 22/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 31/0 : 2[55000] -> 3[77000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 26/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 29/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 23/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 27/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 30/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 24/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 28/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 31/0 : 3[77000] -> 0[9000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 25/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 29/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 26/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 30/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 27/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 31/0 : 1[36000] -> 2[55000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 28/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 29/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 30/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 31/0 : 0[9000] -> 1[36000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Connected all rings
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 00/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 02/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 03/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 04/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 06/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 07/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 08/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 10/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 11/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 12/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 00/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 01/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 14/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 03/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 15/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 04/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 16/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 05/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 18/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 07/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 08/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 19/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 00/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 09/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 20/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 01/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 11/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 22/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 02/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 12/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 01/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 04/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 23/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 13/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 02/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 05/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 24/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 15/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 06/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 03/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 16/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 26/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 08/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 00/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 05/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 17/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 27/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 09/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 01/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 19/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 00/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 06/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 28/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 10/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 03/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 00/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 20/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 01/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 07/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 30/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 04/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 12/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 05/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 13/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 23/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Channel 31/0 : 1[b5000] -> 0[85000] via P2P/IPC comm 0x563c09705ef0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 04/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 03/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 10/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 07/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 24/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 14/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 05/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 04/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 08/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 11/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 01/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 25/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 16/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 06/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 06/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 09/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 02/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 13/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 27/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 17/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 08/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 07/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 11/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 03/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 28/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 18/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 09/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 14/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 12/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 08/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 05/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 29/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 10/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 20/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 13/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 15/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 10/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 06/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Channel 31/0 : 2[d5000] -> 1[b5000] via P2P/IPC comm 0x5612cbe2ebf0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 12/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 21/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 15/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 17/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 11/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 07/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 13/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 22/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 16/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 09/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 12/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 14/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 17/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 24/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 18/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 10/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 16/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 14/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 19/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 25/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 19/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 11/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 17/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 20/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 15/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 26/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 21/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 18/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 21/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 13/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 16/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 28/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 23/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 20/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 14/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 22/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 29/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 18/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 24/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 21/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 15/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 23/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 19/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 25/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 22/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 17/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Channel 30/0 : 3[f5000] -> 2[d5000] via P2P/IPC comm 0x55ab49747fc0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 25/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 20/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 27/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 18/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 24/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 26/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 22/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 28/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 19/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 25/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 27/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 29/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 23/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 21/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 26/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Channel 31/0 : 2[55000] -> 1[36000] via P2P/IPC comm 0x563da6326960 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 29/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 24/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 28/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 22/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 30/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 26/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 29/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 23/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Channel 31/0 : 0[85000] -> 3[f5000] via P2P/IPC comm 0x5631935b6f70 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Channel 30/0 : 3[77000] -> 2[55000] via P2P/IPC comm 0x5602717929b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 25/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 27/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 26/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 28/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 27/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 30/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 29/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Channel 31/0 : 1[36000] -> 0[9000] via P2P/IPC comm 0x5586eae69710 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 30/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Channel 31/0 : 0[9000] -> 3[77000] via P2P/IPC comm 0x556943c272b0 nRanks 04
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO ncclCommInitRank comm 0x5631935b6f70 rank 0 nranks 4 cudaDev 4 nvmlDev 4 busId 85000 commId 0x5152a155dda8145a localSize 480 used 1224919856 bytes on core 64 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO ncclCommInitRank comm 0x5612cbe2ebf0 rank 2 nranks 4 cudaDev 6 nvmlDev 6 busId d5000 commId 0x5152a155dda8145a localSize 480 used 1224919856 bytes on core 99 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:354:7371 [4] NCCL INFO Init timings: rank 0 nranks 4 total 0.72 (kernels 0.00, bootstrap 0.04, allgathers 0.00, topo 0.46, graphs 0.00, connections 0.22, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO ncclCommInitRank comm 0x563c09705ef0 rank 1 nranks 4 cudaDev 5 nvmlDev 5 busId b5000 commId 0x5152a155dda8145a localSize 480 used 1224919856 bytes on core 120 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO ncclCommInitRank comm 0x55ab49747fc0 rank 3 nranks 4 cudaDev 7 nvmlDev 7 busId f5000 commId 0x5152a155dda8145a localSize 480 used 1090702128 bytes on core 88 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:356:7378 [6] NCCL INFO Init timings: rank 2 nranks 4 total 0.71 (kernels 0.00, bootstrap 0.03, allgathers 0.05, topo 0.41, graphs 0.00, connections 0.22, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Connected all trees
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO CC Off, Multi-GPU CC Off, workFifoBytes 4194304
vcjob-2107528843008176130-abc12335-66689-worker-0:355:7373 [5] NCCL INFO Init timings: rank 1 nranks 4 total 0.72 (kernels 0.00, bootstrap 0.04, allgathers 0.02, topo 0.44, graphs 0.00, connections 0.22, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:357:7372 [7] NCCL INFO Init timings: rank 3 nranks 4 total 0.72 (kernels 0.00, bootstrap 0.04, allgathers 0.02, topo 0.44, graphs 0.00, connections 0.22, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 256 | 256
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO 32 coll channels, 32 collnet channels, 0 nvls channels, 32 p2p channels, 16 p2p channels per peer
I1007 01:56:42.208782   354 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 0] NCCL_DEBUG: INFO
I1007 01:56:42.208811   355 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 1] NCCL_DEBUG: INFO
I1007 01:56:42.208811   356 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 2] NCCL_DEBUG: INFO
I1007 01:56:42.208845   357 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 34(MODEL_PARALLEL_GROUP) Rank 3] NCCL_DEBUG: INFO
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO ncclCommInitRank comm 0x556943c272b0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 9000 commId 0x74c3b44357f274b7 localSize 480 used 1111411504 bytes on core 1 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:350:7375 [0] NCCL INFO Init timings: rank 0 nranks 4 total 0.72 (kernels 0.00, bootstrap 0.08, allgathers 0.01, topo 0.46, graphs 0.00, connections 0.17, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO ncclCommInitRank comm 0x563da6326960 rank 2 nranks 4 cudaDev 2 nvmlDev 2 busId 55000 commId 0x74c3b44357f274b7 localSize 480 used 1224919856 bytes on core 32 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO ncclCommInitRank comm 0x5586eae69710 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 36000 commId 0x74c3b44357f274b7 localSize 480 used 1224919856 bytes on core 48 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:352:7379 [2] NCCL INFO Init timings: rank 2 nranks 4 total 0.71 (kernels 0.00, bootstrap 0.07, allgathers 0.01, topo 0.46, graphs 0.00, connections 0.17, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO ncclCommInitRank comm 0x5602717929b0 rank 3 nranks 4 cudaDev 3 nvmlDev 3 busId 77000 commId 0x74c3b44357f274b7 localSize 480 used 1224919856 bytes on core 25 - Init COMPLETE
vcjob-2107528843008176130-abc12335-66689-worker-0:351:7377 [1] NCCL INFO Init timings: rank 1 nranks 4 total 0.72 (kernels 0.00, bootstrap 0.08, allgathers 0.00, topo 0.47, graphs 0.00, connections 0.17, rest 0.00)
vcjob-2107528843008176130-abc12335-66689-worker-0:353:7376 [3] NCCL INFO Init timings: rank 3 nranks 4 total 0.72 (kernels 0.00, bootstrap 0.08, allgathers 0.01, topo 0.46, graphs 0.00, connections 0.17, rest 0.00)
I1007 01:56:42.212450   352 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 2] NCCL_DEBUG: INFO
I1007 01:56:42.212452   351 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 1] NCCL_DEBUG: INFO
I1007 01:56:42.212466   350 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 0] NCCL_DEBUG: INFO
I1007 01:56:42.212527   353 ProcessGroupNCCL.cpp:2890] [PG ID 4 PG GUID 33(MODEL_PARALLEL_GROUP) Rank 3] NCCL_DEBUG: INFO
Number of parameters in transformer block in billions:  7.78
Number of active parameters in transformer block in billions:  7.78
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 8.05
Total number of active parameters in billions: 8.05
Number of parameters in most loaded shard in billions: 2.0125
compute_activation_memory_without_sp
Activation memory footprint per transformer layer (precise, without SP): 256.0 MB
Theoretical memory footprints: weight and optimizer=17273.29 MB, activation=8718.18 MB, total=25991.47 MB
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
[Rank 3] (after 1 iterations) memory (MB) | allocated: 17681.27 | max allocated: 17681.27 | reserved: 19554.00 | max reserved: 19554.00[Rank 2] (after 1 iterations) memory (MB) | allocated: 17681.27 | max allocated: 17681.27 | reserved: 19554.00 | max reserved: 19554.00[Rank 1] (after 1 iterations) memory (MB) | allocated: 17681.27 | max allocated: 17681.27 | reserved: 19546.00 | max reserved: 19546.00
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
[Rank 0] (after 1 iterations) memory (MB) | allocated: 17697.38 | max allocated: 17697.38 | reserved: 19554.00 | max reserved: 19554.00
Number of parameters in transformer block in billions:  7.78
Number of active parameters in transformer block in billions:  7.78
Number of parameters in embedding layers in billions: 0.26
Total number of parameters in billions: 8.05
Total number of active parameters in billions: 8.05
Number of parameters in most loaded shard in billions: 2.0125
compute_activation_memory_without_sp
Activation memory footprint per transformer layer (precise, without SP): 256.0 MB
Theoretical memory footprints: weight and optimizer=17273.29 MB, activation=8718.18 MB, total=25991.47 MB
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
[Rank 2] (after 2 iterations) memory (MB) | allocated: 17681.27 | max allocated: 18221.44 | reserved: 20144.00 | max reserved: 20144.00
[Rank 3] (after 2 iterations) memory (MB) | allocated: 17681.27 | max allocated: 18221.44 | reserved: 20144.00 | max reserved: 20144.00[Rank 1] (after 2 iterations) memory (MB) | allocated: 17681.27 | max allocated: 18221.44 | reserved: 20264.00 | max reserved: 20264.00
missing _msg field; see https://docs.victoriametrics.com/victorialogs/keyconcepts/#message-field
[Rank 0] (after 2 iterations) memory (MB) | allocated: 17697.38 | max allocated: 18221.55 | reserved: 20144.00 | max reserved: 20144.00
 至底部