import ChatMessage from "@/types/message";
import { Avatar, Sheet, Skeleton, Typography } from "@mui/joy";

export default function Bubble({ message }: { message: ChatMessage }) {
  const { agent, content, loading } = message;

  return (
    <Sheet
      sx={{
        display: "flex",
        flexDirection: agent ? "row" : "row-reverse",
        margin: 1,
      }}
    >
      {agent?.avatar_url && (
        <Avatar size="lg" src={`/api${agent.avatar_url}`} sx={{ mx: 2 }} />
      )}
      <div>
        <Typography level="body-sm">{agent?.name}</Typography>
        <Sheet
          variant={agent ? "soft" : "solid"}
          color={agent ? "neutral" : "primary"}
          sx={{
            width: loading ? 125 : undefined,
            maxWidth: 250,
            borderRadius: "md",
            p: 2,
          }}
        >
          {loading ? (
            <div>
              <Skeleton variant="text" width="100%" loading={loading} />
              <Skeleton variant="text" width="75%" loading={loading} />
            </div>
          ) : (
            <Typography textColor={agent ? undefined : "white"}>
              {content}
            </Typography>
          )}
        </Sheet>
      </div>
    </Sheet>
  );
}
