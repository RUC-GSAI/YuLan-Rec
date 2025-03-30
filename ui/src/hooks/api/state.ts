import SystemState from "@/types/api/state/system";
import useSWR from "swr";

export function useSystemState() {
  const { data, error, isLoading } = useSWR<SystemState>("/api/system-stat", {
    refreshInterval: 10 * 1000,
  });

  return {
    state: data,
    error,
    isLoading,
  };
}
