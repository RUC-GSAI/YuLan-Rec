import AgentCard from "@/components/AgentCard";
import ButtonPanel from "@/components/ButtonPanel";
import ChatCard from "@/components/ChatCard";
import Header from "@/components/Header";
import LogCard from "@/components/LogCard";
import StateCard from "@/components/StateCard";
import {
  DisplayedAgentContext,
  DisplayedAgentDispatchContext,
} from "@/context/DisplayedAgentContext";
import { Box, Sheet } from "@mui/joy";
import { lazy, useState } from "react";

const Graph = lazy(() => import("@/components/Graph"));

export default function Home() {
  const [displayedUser, setDisplayedUser] = useState(0);

  return (
    <DisplayedAgentContext.Provider value={displayedUser}>
      <DisplayedAgentDispatchContext.Provider value={setDisplayedUser}>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            width: "100vw",
            height: "100vh",
          }}
        >
          <Header />
          <Sheet variant="soft" sx={{ display: "flex", flexGrow: 1 }}>
            <Box
              sx={{
                m: 2,
                mr: 0,
                width: 320,
                display: "flex",
                flexDirection: "column",
              }}
            >
              <AgentCard sx={{ mb: 1, flexGrow: "1" }} />
              <ChatCard key={displayedUser} sx={{ mt: 1 }} />
            </Box>
            <Box
              sx={{
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                flexGrow: 1,
              }}
            >
              <Graph />
              <ButtonPanel sx={{ position: "absolute", bottom: 0, mb: 2 }} />
            </Box>
            <Box
              sx={{
                m: 2,
                ml: 0,
                width: 320,
                display: "flex",
                flexDirection: "column",
              }}
            >
              <StateCard sx={{ mb: 1 }} />
              <LogCard sx={{ mt: 1, flexGrow: 1 }} />
            </Box>
          </Sheet>
        </Box>
      </DisplayedAgentDispatchContext.Provider>
    </DisplayedAgentContext.Provider>
  );
}
