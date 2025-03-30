import { createContext } from "react";

export const DisplayedAgentContext = createContext(0);
export const DisplayedAgentDispatchContext = createContext<
  React.Dispatch<number>
>(() => {});
