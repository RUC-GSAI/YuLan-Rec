export default interface Event {
  action_type: "idle" | "watching" | "chatting" | "posting";
  start_time: Date;
  end_time: Date;
  duration: number;
  target_agent?: number;
}
