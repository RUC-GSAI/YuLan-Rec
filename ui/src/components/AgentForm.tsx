import Agent from "@/types/api/agent";
import {
  Button,
  Card,
  CardActions,
  CardContent,
  FormControl,
  FormLabel,
  Input,
  Modal,
  ModalClose,
  Option,
  Select,
  Textarea,
  Typography,
} from "@mui/joy";
import axios, { HttpStatusCode } from "axios";
import { toast } from "sonner";
import { useImmer } from "use-immer";

export default function AgentForm({
  open,
  setOpen,
  initialAgent,
  mutateInitialAgent,
}: {
  open: boolean;
  setOpen: (open: boolean) => void;
  initialAgent: Agent;
  mutateInitialAgent: () => void;
}) {
  const [agent, updateAgent] =
    useImmer<Omit<Agent, "avatar_url" | "event">>(initialAgent);

  async function handleSave() {
    const res = await axios.put(`/api/agents/${agent.id}`, agent);
    if (res.status == HttpStatusCode.Ok) {
      mutateInitialAgent();
      toast.success("Agent updated");
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
        <Typography level="title-lg">Edit Agent</Typography>
        <ModalClose />
        <CardContent
          sx={{
            overflowY: "auto",
            overflowX: "hidden",
          }}
        >
          <FormControl>
            <FormLabel>Name</FormLabel>
            <Input
              size="sm"
              value={agent.name}
              onChange={(e) => {
                updateAgent((draft) => {
                  draft.name = e.target.value;
                });
              }}
            />
          </FormControl>

          <FormControl>
            <FormLabel>Gender</FormLabel>
            <Select
              size="sm"
              value={agent.gender}
              onChange={(_, value) => {
                updateAgent((draft) => {
                  if (value) {
                    draft.gender = value;
                  }
                });
              }}
            >
              <Option value="female">Female</Option>
              <Option value="male">Male</Option>
            </Select>
          </FormControl>

          <FormControl>
            <FormLabel>Age</FormLabel>
            <Input
              size="sm"
              type="number"
              value={agent.age}
              onChange={(e) => {
                updateAgent((draft) => {
                  draft.age = parseInt(e.target.value);
                });
              }}
            />
          </FormControl>

          <FormControl>
            <FormLabel>Status</FormLabel>
            <Input
              size="sm"
              value={agent.status}
              onChange={(e) => {
                updateAgent((draft) => {
                  draft.status = e.target.value;
                });
              }}
            />
          </FormControl>

          <FormControl>
            <FormLabel>Traits</FormLabel>
            <Textarea
              size="sm"
              minRows={2}
              value={agent.traits}
              onChange={(e) => {
                updateAgent((draft) => {
                  draft.traits = e.target.value;
                });
              }}
            />
          </FormControl>

          <FormControl>
            <FormLabel>Interest</FormLabel>
            <Textarea
              size="sm"
              minRows={2}
              value={agent.interest}
              onChange={(e) => {
                updateAgent((draft) => {
                  draft.interest = e.target.value;
                });
              }}
            />
          </FormControl>

          <FormControl>
            <FormLabel>Feature</FormLabel>
            <Textarea
              size="sm"
              minRows={2}
              value={agent.feature}
              onChange={(e) => {
                updateAgent((draft) => {
                  draft.feature = e.target.value;
                });
              }}
            />
          </FormControl>
        </CardContent>

        <CardActions buttonFlex="1">
          <Button onClick={handleSave}>Save</Button>
        </CardActions>
      </Card>
    </Modal>
  );
}
