import Event from "./event";

export default interface Agent {
  id: number;
  name: string;
  avatar_url: string;
  idle_url: string;
  watching_url: string;
  chatting_url: string;
  posting_url: string;
  gender: "female" | "male";
  age: number;
  traits: string;
  status: string;
  interest: string;
  feature: string;
  event: Event;
}
