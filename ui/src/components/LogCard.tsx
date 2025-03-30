import { useMessages } from "@/hooks/api";
import LogMessage from "@/types/api/message";
import { CleaningServices } from "@mui/icons-material";
import {
  Card,
  CardContent,
  IconButton,
  Sheet,
  Tooltip,
  Typography,
} from "@mui/joy";
import { SxProps } from "@mui/joy/styles/types";
import { useEffect, useRef } from "react";
import { useImmer } from "use-immer";

export default function LogCard({ sx }: { sx?: SxProps }) {
  const { messages: newMessages } = useMessages();
  const [messages, updateMessages] = useImmer<LogMessage[]>([]);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (newMessages && newMessages.length !== 0) {
      updateMessages((draft) => {
        draft.push(...newMessages);
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [newMessages]);

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  function handleClean() {
    updateMessages([]);
  }

  return (
    <Card variant="outlined" sx={{ ...sx, height: 100 }}>
      <Sheet
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Typography level="title-lg">Log Messages</Typography>
        <Tooltip variant="soft" title="Clean all">
          <IconButton onClick={handleClean}>
            <CleaningServices />
          </IconButton>
        </Tooltip>
      </Sheet>

      <CardContent sx={{ height: 100 }}>
        <Sheet
          ref={messagesContainerRef}
          variant="soft"
          sx={{
            height: "100%",
            p: 1,
            borderRadius: "md",
            overflowX: "hidden",
            overflowY: "auto",
          }}
        >
          {messages.map((message, index) => {
            return (
              <Typography key={index} fontFamily="monospace" level="body-sm">
                <Typography variant="solid" color="primary">
                  {message.action}
                </Typography>
                <Typography>: {message.content}</Typography>
              </Typography>
            );
          })}
        </Sheet>
      </CardContent>
    </Card>
  );
}
