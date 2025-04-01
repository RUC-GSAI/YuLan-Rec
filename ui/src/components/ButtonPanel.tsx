import { Settings } from "@mui/icons-material";
import { Button, ButtonGroup, IconButton, Sheet, Tooltip } from "@mui/joy";
import { SxProps } from "@mui/joy/styles/types";
import axios from "axios";
import { useState } from "react";
import { toast } from "sonner";
import SettingsForm from "./SettingsForm";

export default function ButtonPanel({ sx }: { sx?: SxProps }) {
  const [startIsLoading, setStartIsLoading] = useState(false);
  const [pauseIsLoading, setPauseIsLoading] = useState(false);
  const [resetIsLoading, setResetIsLoading] = useState(false);

  async function handleStart() {
    setStartIsLoading(true);
    await axios.post("/api/start");
    setStartIsLoading(false);
    toast.success("Started");
  }

  async function handlePause() {
    setPauseIsLoading(true);
    await axios.post("/api/pause");
    setPauseIsLoading(false);
    toast.success("Paused");
  }

  async function handleReset() {
    setResetIsLoading(true);
    await axios.post("/api/reset");
    setResetIsLoading(false);
    toast.success("Reset");
  }

  const [openSettings, setOpenSettings] = useState(false);

  return (
    <Sheet sx={{ ...sx }}>
      <ButtonGroup size="lg" variant="outlined">
        <Button disabled={startIsLoading} onClick={handleStart}>
          Start
        </Button>
        <Button disabled={pauseIsLoading} onClick={handlePause}>
          Pause
        </Button>
        <Button disabled={resetIsLoading} onClick={handleReset}>
          Reset
        </Button>
        <Tooltip title="Open Settings" variant="soft">
          <IconButton onClick={() => setOpenSettings(true)}>
            <Settings />
          </IconButton>
        </Tooltip>
      </ButtonGroup>

      <SettingsForm open={openSettings} setOpen={setOpenSettings} />
    </Sheet>
  );
}
