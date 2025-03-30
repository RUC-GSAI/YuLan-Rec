import UiSettings from "@/types/settings";
import { createContext } from "react";

export const UiSettingsContext = createContext<UiSettings>({
  node_size: 20,
  active_node_size: 40,
  background_image: "",
});
export const UiSettingsDispatchContext = createContext<
  React.Dispatch<UiSettings>
>(() => {});
