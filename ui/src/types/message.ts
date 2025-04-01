import Agent from "./api/agent";

export default interface ChatMessage {
  id: number;
  content: string;
  //HACK: System agent is a boolean value
  agent?: Agent | boolean;
  loading: boolean;
}
