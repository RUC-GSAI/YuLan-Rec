import LogMessage from "@/types/api/message";
import AgentSettings from "@/types/api/setting";
import useSWR from "swr";

export function useRound() {
  const { data, error, isLoading } = useSWR<number>("/api/rounds", {
    refreshInterval: 10 * 1000,
  });

  return {
    round: data,
    error,
    isLoading,
  };
}

export function useAgentSettings() {
  const { data, error, isLoading, mutate } =
    useSWR<AgentSettings>("/api/configs");

  return {
    agentSettings: data,
    error,
    isLoading,
    mutate,
  };
}

export function useMessages() {
  const { data, error, isLoading } = useSWR<LogMessage[]>("/api/messages", {
    refreshInterval: 10 * 1000,
  });

  return {
    messages: data,
    error,
    isLoading,
  };
}
