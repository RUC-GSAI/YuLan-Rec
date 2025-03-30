import { useRound } from "@/hooks/api";
import {
  DarkModeRounded,
  LightModeRounded,
  Menu,
  SearchRounded,
} from "@mui/icons-material";
import { Box, IconButton, Typography, useColorScheme } from "@mui/joy";
import { useEffect, useState } from "react";
import SearchBox from "./SearchBox";

function ColorSchemeToggle() {
  const { mode, setMode } = useColorScheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);
  if (!mounted) {
    return <IconButton size="sm" variant="soft" color="neutral" />;
  }
  return (
    <IconButton
      id="toggle-mode"
      size="sm"
      variant="soft"
      color="neutral"
      onClick={() => {
        if (mode === "light") {
          setMode("dark");
        } else {
          setMode("light");
        }
      }}
    >
      {mode === "light" ? <DarkModeRounded /> : <LightModeRounded />}
    </IconButton>
  );
}

export default function Header() {
  const { round } = useRound();

  return (
    <Box
      component="header"
      className="Header"
      sx={{
        p: 1,
        gap: 2,
        bgcolor: "background.surface",
        display: "flex",
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        gridColumn: "1 / -1",
        borderBottom: "1px solid",
        borderColor: "divider",
        position: "sticky",
        zIndex: 100,
        top: 0,
      }}
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          gap: 1.5,
        }}
      >
        <IconButton
          variant="outlined"
          size="sm"
          // onClick={() => setDrawerOpen(true)}
          sx={{ display: { sm: "none" } }}
        >
          <Menu />
        </IconButton>
        <IconButton
          size="sm"
          variant="soft"
          sx={{ display: { xs: "none", sm: "inline-flex" } }}
        >
          <img src="/logo.svg" alt="logo" width="24" height="24" />
        </IconButton>
        <Typography component="h1" fontWeight="xl">
          RecAgent
        </Typography>
        <Typography level="body-sm">Round {round}</Typography>
      </Box>
      <SearchBox />
      <Box sx={{ display: "flex", flexDirection: "row", gap: 1.5 }}>
        <IconButton
          size="sm"
          variant="outlined"
          color="neutral"
          sx={{ display: { xs: "inline-flex", sm: "none" } }}
        >
          <SearchRounded />
        </IconButton>

        <ColorSchemeToggle />
      </Box>
    </Box>
  );
}
