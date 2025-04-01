import { DisplayedAgentContext } from "@/context/DisplayedAgentContext";
import { useAgent, usePlayingAgent } from "@/hooks/api/agent";
import ChatMessage from "@/types/message";
import { ExpandLess, ExpandMore, Send } from "@mui/icons-material";
import {
  Card,
  CardActions,
  CardContent,
  IconButton,
  Input,
  Sheet,
  Tooltip,
  Typography,
} from "@mui/joy";
import { SxProps } from "@mui/joy/styles/types";
import axios, { HttpStatusCode } from "axios";
import { useContext, useEffect, useRef, useState } from "react";
import { useImmer } from "use-immer";
import Bubble from "./Bubble";

export default function ChatCard({ sx }: { sx?: SxProps }) {
  const displayedAgentId = useContext(DisplayedAgentContext);

  const [contentVisible, setContentVisible] = useState(false);

  const { agent } = useAgent(displayedAgentId);
  const { id: playingAgentId } = usePlayingAgent();

  const messagesCounter = useRef(0);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const [messages, setMessages] = useImmer<ChatMessage[]>([]);
  const [request, setRequest] = useState("");
  const [loading, setLoading] = useState(false);

  const protocol = useRef<"http" | "ws">("http");

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (agent && agent.id === playingAgentId) {
      protocol.current = "ws";
      setLoading(true);
      ws.current = new WebSocket(
        `ws://${process.env.VITE_API_ADDRESS}/role-play/${playingAgentId}`,
      );
      ws.current.addEventListener("open", () => {
        setLoading(false);
      });

      ws.current.addEventListener("message", (e) => {
        setMessages((draft) => {
          draft.push({
            id: messagesCounter.current,
            agent: true,
            loading: false,
            content: e.data,
          });
        });
        messagesCounter.current++;
      });

      return () => {
        ws.current?.close();
      };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent, playingAgentId, ws]);

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop =
        messagesContainerRef.current.scrollHeight;
    }
  }, [messages]);

  async function handleSend() {
    if (!agent) return;
    if (protocol.current === "http") {
      setLoading(true);
      setMessages((draft) => {
        draft.push({
          id: messagesCounter.current,
          loading: false,
          content: request,
        });
        draft.push({
          id: messagesCounter.current + 1,
          loading: true,
          agent,
          content: "...",
        });
        messagesCounter.current += 2;
      });
      setRequest("");
      const res = await axios.get(
        `/api/interview-agents/${agent.id}?query=${request}`,
      );
      if (res.status === HttpStatusCode.Ok) {
        setMessages((draft) => {
          draft.pop();
          draft.push({
            id: messagesCounter.current + 1,
            loading: false,
            agent,
            content: res.data,
          });
        });
        setLoading(false);
      }
    } else if (protocol.current === "ws") {
      setMessages((draft) => {
        draft.push({
          id: messagesCounter.current,
          loading: false,
          content: request,
        });
        messagesCounter.current++;
      });
      setRequest("");
      ws.current?.send(request);
    }
  }

  return (
    <Card variant="outlined" sx={{ ...sx }}>
      <Sheet
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Typography level="title-lg">
          {protocol.current === "http" ? "Interview" : "Role Play"}
        </Typography>
        <Tooltip title="Fold panel" variant="soft">
          <IconButton
            size="sm"
            onClick={() => setContentVisible((prev) => !prev)}
          >
            {contentVisible ? <ExpandMore /> : <ExpandLess />}
          </IconButton>
        </Tooltip>
      </Sheet>

      {contentVisible && (
        <CardContent
          ref={messagesContainerRef}
          sx={{
            overflowY: "auto",
            overflowX: "hidden",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
          }}
        >
          <Sheet
            sx={{
              minHeight: 200,
              maxHeight: 500,
            }}
          >
            {messages.length === 0 ? (
              <Typography
                level="body-md"
                color="neutral"
                fontStyle="italic"
                sx={{ textAlign: "center" }}
              >
                Start a conversation
              </Typography>
            ) : (
              messages.map((message) => (
                <Bubble key={message.id} message={message} />
              ))
            )}
          </Sheet>
        </CardContent>
      )}

      {contentVisible && (
        <CardActions component="form">
          <Input
            sx={{ flex: 1 }}
            value={request}
            onChange={(e) => setRequest(e.target.value)}
          />
          <Tooltip title="Send message" variant="soft">
            <IconButton
              type="submit"
              color="primary"
              variant="solid"
              disabled={loading || !request}
              onClick={(e) => {
                e.preventDefault();
                handleSend();
              }}
            >
              <Send />
            </IconButton>
          </Tooltip>
        </CardActions>
      )}
    </Card>
  );
}
