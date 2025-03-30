import { DisplayedAgentContext } from "@/context/DisplayedAgentContext";
import { useAgent, usePlayingAgent } from "@/hooks/api/agent";
import { Edit, Female, Male } from "@mui/icons-material";
import {
  Avatar,
  Card,
  CardContent,
  Divider,
  IconButton,
  Sheet,
  Skeleton,
  Tooltip,
  Typography,
} from "@mui/joy";
import { SxProps } from "@mui/joy/styles/types";
import { useContext, useState } from "react";
import { capitalizeFirstLetter, stringToColor } from "../utils";
import AgentForm from "./AgentForm";

function stringAvatar(name: string) {
  return {
    sx: {
      bgcolor: stringToColor(name),
    },
    children: `${name.split(" ")[0][0]}${name.split(" ")[1][0]}`,
  };
}

export default function AgentCard({ sx }: { sx?: SxProps }) {
  const displayedAgentId = useContext(DisplayedAgentContext);

  const { agent, mutate: mutateAgent } = useAgent(displayedAgentId);
  const { id: playingAgentId } = usePlayingAgent();

  const [openEdit, setOpenEdit] = useState(false);

  return (
    <Card variant="outlined" sx={{ ...sx, height: 100 }}>
      <Sheet sx={{ display: "flex", alignItems: "center" }}>
        {agent ? (
          !agent.avatar_url || agent.avatar_url == "" ? (
            <Avatar size="lg" {...stringAvatar(agent.name)} />
          ) : (
            <Avatar size="lg" src={"/api" + agent.avatar_url} />
          )
        ) : (
          <Avatar size="lg" src="">
            <Skeleton />
          </Avatar>
        )}

        <Sheet sx={{ ml: 2, flexGrow: "1" }}>
          {agent ? (
            <>
              <Sheet
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <Sheet sx={{ display: "flex", alignItems: "center" }}>
                  <Typography noWrap level="title-lg">
                    {agent.name}
                  </Typography>
                  {agent.gender == "male" ? (
                    <Male sx={{ color: "blue" }} />
                  ) : (
                    <Female sx={{ color: "pink" }} />
                  )}
                  {agent.id === playingAgentId && (
                    <Typography
                      noWrap
                      level="body-sm"
                      variant="soft"
                      color="success"
                    >
                      Playing
                    </Typography>
                  )}
                </Sheet>
                <Tooltip title="Edit agent" variant="soft">
                  <IconButton
                    variant="soft"
                    size="sm"
                    onClick={() => setOpenEdit(true)}
                  >
                    <Edit />
                  </IconButton>
                </Tooltip>
              </Sheet>
              <Sheet
                sx={{
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <Typography level="title-md" sx={{ marginRight: 2 }}>
                  {capitalizeFirstLetter(agent.status)}
                </Typography>
                <Typography level="title-md">{agent.age} years old</Typography>
              </Sheet>
            </>
          ) : (
            <>
              <Skeleton variant="text" level="title-lg" width="50%" />
              <Skeleton variant="text" level="title-md" width="80%" />
            </>
          )}
        </Sheet>
      </Sheet>

      <CardContent
        sx={{
          height: 100,
          overflowX: "hidden",
          overflowY: "auto",
        }}
      >
        {agent ? (
          <>
            <Sheet sx={{ mb: 2 }}>
              <Divider>Traits</Divider>
              <Typography>{agent.traits}</Typography>
            </Sheet>

            <Sheet sx={{ mb: 2 }}>
              <Divider>Interest</Divider>
              <Typography>{agent.interest}</Typography>
            </Sheet>

            <Sheet sx={{ mb: 2 }}>
              <Divider>Feature</Divider>
              <Typography>{agent.feature}</Typography>
            </Sheet>
          </>
        ) : (
          <>
            <Skeleton variant="text" width="90%" />
            <Skeleton variant="text" />
            <Skeleton variant="text" width="95%" />
            <Skeleton variant="text" width="90%" />
            <Skeleton variant="text" />
            <Skeleton variant="text" width="95%" />
          </>
        )}
      </CardContent>

      {agent && (
        <AgentForm
          open={openEdit}
          setOpen={setOpenEdit}
          initialAgent={agent}
          mutateInitialAgent={mutateAgent}
        />
      )}
    </Card>
  );
}
