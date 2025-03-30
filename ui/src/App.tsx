import { CssVarsProvider, GlobalStyles } from "@mui/joy";
import { Toaster } from "sonner";
import { SWRConfig } from "swr";
import { useImmer } from "use-immer";
import {
  UiSettingsContext,
  UiSettingsDispatchContext,
} from "./context/UiSettingsContext";
import Home from "./pages/Home";
import UiSettings from "./types/settings";
import { fetcher } from "./utils/fecher";

export default function App() {
  const [uiSettings, updateUiSettings] = useImmer<UiSettings>({
    node_size: 30,
    active_node_size: 60,
  });

  return (
    <SWRConfig value={{ fetcher: fetcher }}>
      <CssVarsProvider>
        <UiSettingsContext.Provider value={uiSettings}>
          <UiSettingsDispatchContext.Provider value={updateUiSettings}>
            <GlobalStyles
              styles={(theme) => `
        [data-sonner-toaster][data-theme] {
          font-family: ${theme.vars.fontFamily.body};
          font-size: ${theme.fontSize.md};
          --border-radius: ${theme.vars.radius.sm};
          --normal-bg: ${theme.vars.palette.background.surface};
          --normal-border: ${theme.vars.palette.divider};
          --normal-text: ${theme.vars.palette.text.primary};
          --success-bg: ${theme.vars.palette.success.softBg};
          --success-border: rgb(${theme.vars.palette.success.mainChannel} / 0.2);
          --success-text: ${theme.vars.palette.success.softColor};
          --error-bg: ${theme.vars.palette.danger.softBg};
          --error-border: rgb(${theme.vars.palette.danger.mainChannel} / 0.2);
          --error-text: ${theme.vars.palette.danger.softColor};
          --gray1: ${theme.vars.palette.neutral[50]};
          --gray2: ${theme.vars.palette.neutral[100]};
          --gray3: ${theme.vars.palette.neutral[200]};
          --gray4: ${theme.vars.palette.neutral[300]};
          --gray5: ${theme.vars.palette.neutral[400]};
          --gray6: ${theme.vars.palette.neutral[500]};
          --gray7: ${theme.vars.palette.neutral[600]};
          --gray8: ${theme.vars.palette.neutral[700]};
          --gray9: ${theme.vars.palette.neutral[800]};
          --gray10: ${theme.vars.palette.neutral[900]};
        }
        &.sonner-toast-warn {
          --normal-bg: ${theme.vars.palette.warning.softBg};
          --normal-border: rgb(${theme.vars.palette.warning.mainChannel} / 0.2);
          --normal-text: ${theme.vars.palette.warning.softColor};
        }
      `}
            />
            <Toaster position="top-center" richColors closeButton />
            <Home />
          </UiSettingsDispatchContext.Provider>
        </UiSettingsContext.Provider>
      </CssVarsProvider>
    </SWRConfig>
  );
}
