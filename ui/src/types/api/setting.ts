export interface UiSettings {
  node_size: number;
  active_node_size: number;
}

export interface AgentSettings {
  output_file: string;
  log_file: string;
  log_name: string;
  play_role: boolean;
  item_path: string;
  user_path: string;
  relationship_path: string;
  interaction_path: string;
  index_name: string;
  simulator_dir: string;
  simulator_restore_file_name: string;
  model: string;
  epoch: number;
  agent_num: number;
  page_size: number;
  temperature: number;
  max_token: number;
  execution_mode: string;
  interval: string;
  llm: string;
  max_retries: number;
  verbose: boolean;
  active_agent_threshold: number;
  active_method: string;
  active_prob: number;
  recagent_memory: string;
  api_keys: string[];
}
