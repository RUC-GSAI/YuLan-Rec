import Relationship from "@/types/api/relationship";
import useSWR from "swr";

export function useRelationship() {
  const { data, error, isLoading } =
    useSWR<Relationship[]>("/api/relationships");

  return {
    relationship: data,
    error,
    isLoading,
  };
}
