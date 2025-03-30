import Agent from "@/types/api/agent";
import useSWR from "swr";

export function useAgents() {
  const { data, error, isLoading } = useSWR<Agent[]>("/api/agents");

  return {
    agents: data,
    error,
    isLoading,
  };
}

export function useActiveAgents() {
  const { data, error, isLoading } = useSWR<Agent[]>("/api/active-agents", {
    refreshInterval: 10 * 1000,
  });

  return {
    agents: data,
    error,
    isLoading,
  };
}

export function useAgent(id: number) {
  const { data, error, isLoading, mutate } = useSWR<Agent>(`/api/agents/${id}`);

  return {
    agent: data,
    error,
    isLoading,
    mutate,
  };
}

export function useQueryAgents(query: string) {
  const { data, error, isLoading } = useSWR<Agent[]>(
    `/api/agents?query=${query}`,
  );

  return {
    agents: data,
    error,
    isLoading,
  };
}

export function usePlayingAgent() {
  const { data, error, isLoading } = useSWR<number>("/api/role-play");

  return {
    id: data,
    error,
    isLoading,
  };
}
