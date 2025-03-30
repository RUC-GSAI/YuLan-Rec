import {
  UiSettingsContext,
  UiSettingsDispatchContext,
} from "@/context/UiSettingsContext";
import { useAgentSettings } from "@/hooks/api";
import AgentSettings from "@/types/api/setting";
import { Refresh } from "@mui/icons-material";
import {
  Button,
  Card,
  CardActions,
  CardContent,
  Divider,
  FormControl,
  FormHelperText,
  FormLabel,
  IconButton,
  Input,
  Modal,
  ModalClose,
  Switch,
  Textarea,
  Tooltip,
  Typography,
} from "@mui/joy";
import axios, { HttpStatusCode } from "axios";
import { useContext, useEffect } from "react";
import { toast } from "sonner";
import { useImmer } from "use-immer";

export default function SettingsForm({
  open,
  setOpen,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
}) {
  const initialUiSettings = useContext(UiSettingsContext);
  const updateInitialUiSettings = useContext(UiSettingsDispatchContext);
  const [uiSettings, updateUiSettings] = useImmer(initialUiSettings);

  const [agentSettings, updateAgentSettings] = useImmer<AgentSettings>({
    output_file: "",
    log_file: "",
    log_name: "",
    play_role: false,
    item_path: "",
    user_path: "",
    relationship_path: "",
    interaction_path: "",
    index_name: "",
    simulator_dir: "",
    simulator_restore_file_name: "",
    model: "",
    epoch: 0,
    agent_num: 0,
    page_size: 0,
    temperature: 0,
    max_token: 0,
    execution_mode: "",
    interval: "",
    llm: "",
    max_retries: 0,
    verbose: false,
    active_agent_threshold: 0,
    active_method: "",
    active_prob: 0,
    recagent_memory: "",
    api_keys: [],
  });

  const {
    agentSettings: initialAgentSettings,
    mutate: mutateInitialAgentSettings,
  } = useAgentSettings();

  useEffect(() => {
    if (initialAgentSettings) {
      updateAgentSettings(initialAgentSettings);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialAgentSettings]);

  function handleRefresh() {
    updateUiSettings(initialUiSettings);
    if (initialAgentSettings) {
      updateAgentSettings(initialAgentSettings);
    }
    toast.success("Reset properties");
  }

  async function handleSave() {
    updateInitialUiSettings(uiSettings);
    const res = await axios.patch("/api/configs", agentSettings);
    if (res.status === HttpStatusCode.Ok) {
      mutateInitialAgentSettings();
      toast.success("Save successfully");
    }
  }

  return (
    <Modal
      open={open}
      onClose={() => setOpen(false)}
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Card variant="outlined" sx={{ width: 400, height: 400 }}>
        <Typography level="title-lg">Configure Settings</Typography>
        <ModalClose />
        <CardContent
          sx={{
            overflowY: "auto",
            overflowX: "hidden",
          }}
        >
          <Divider>UI Settings</Divider>
          <FormControl>
            <FormLabel>Node Size</FormLabel>
            <Input
              size="sm"
              type="number"
              value={uiSettings.node_size}
              onChange={(e) => {
                updateUiSettings((draft) => {
                  draft.node_size = parseInt(e.target.value);
                });
              }}
            />
          </FormControl>

          <FormControl>
            <FormLabel>Active Node Size</FormLabel>
            <Input
              size="sm"
              type="number"
              value={uiSettings.active_node_size}
              onChange={(e) => {
                updateUiSettings((draft) => {
                  draft.active_node_size = parseInt(e.target.value);
                });
              }}
            />
          </FormControl>

          {initialAgentSettings && (
            <>
              <Divider>Agent Settings</Divider>
              <FormControl>
                <FormLabel>Output File</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.output_file}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.output_file = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Log File</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.log_file}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.log_file = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Log Name</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.log_name}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.log_name = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Play Role</FormLabel>
                <Switch
                  checked={agentSettings.play_role}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.play_role = e.target.checked;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Item Path</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.item_path}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.item_path = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>User Path</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.user_path}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.user_path = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Relationship Path</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.relationship_path}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.relationship_path = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Interaction Path</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.interaction_path}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.interaction_path = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Index Name</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.index_name}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.index_name = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Simulator Directory</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.simulator_dir}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.simulator_dir = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Simulator Restore File Name</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.simulator_restore_file_name}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.simulator_restore_file_name = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Model</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.model}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.model = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Epoch</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.epoch}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.epoch = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Agent number</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.agent_num}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.agent_num = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Page Size</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.page_size}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.page_size = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Temperature</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.temperature}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.temperature = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Max Token</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.max_token}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.max_token = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Execution Mode</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.execution_mode}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.execution_mode = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Interval</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.interval}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.interval = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>LLM</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.llm}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.llm = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Max Retries</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.max_retries}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.max_retries = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Verbose</FormLabel>
                <Switch
                  checked={agentSettings.verbose}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.verbose = e.target.checked;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Active Agent Threshold</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.active_agent_threshold}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.active_agent_threshold = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Active Method</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.active_method}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.active_method = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Active Probability</FormLabel>
                <Input
                  size="sm"
                  type="number"
                  value={agentSettings.active_prob}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.active_prob = parseInt(e.target.value);
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Recagent Memory</FormLabel>
                <Input
                  size="sm"
                  value={agentSettings.recagent_memory}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.recagent_memory = e.target.value;
                    });
                  }}
                />
              </FormControl>

              <FormControl>
                <FormLabel>API Keys</FormLabel>
                <Textarea
                  size="sm"
                  minRows={2}
                  value={agentSettings.api_keys.join(",")}
                  onChange={(e) => {
                    updateAgentSettings((draft) => {
                      draft.api_keys = e.target.value
                        .split(",")
                        .map((v) => v.trim());
                    });
                  }}
                />
                <FormHelperText>Split with comma</FormHelperText>
              </FormControl>
            </>
          )}
        </CardContent>

        <CardActions buttonFlex="1">
          <Button onClick={handleSave}>Save</Button>
          <Tooltip title="Reset properties" variant="soft">
            <IconButton variant="outlined" onClick={handleRefresh}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </CardActions>
      </Card>
    </Modal>
  );
}
