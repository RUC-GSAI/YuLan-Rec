import { DisplayedAgentDispatchContext } from "@/context/DisplayedAgentContext";
import { useQueryAgents } from "@/hooks/api/agent";
import { Female, Male, SearchRounded } from "@mui/icons-material";
import {
  Autocomplete,
  AutocompleteOption,
  Avatar,
  Box,
  CircularProgress,
  ListItemContent,
  ListItemDecorator,
  Typography,
} from "@mui/joy";
import { useContext, useState } from "react";

export default function SearchBox() {
  const [query, setQuery] = useState(" ");
  const { agents, isLoading } = useQueryAgents(query);

  const setDisplayedAgentId = useContext(DisplayedAgentDispatchContext);

  return (
    <Autocomplete
      options={agents ?? []}
      getOptionLabel={(option) => option.name}
      isOptionEqualToValue={(option, value) => option.id === value.id}
      renderOption={(props, agent) => (
        <AutocompleteOption {...props} key={agent.id}>
          <ListItemDecorator>
            <Avatar src={"/api" + agent.avatar_url} sx={{ mr: 1 }} />
          </ListItemDecorator>
          <ListItemContent>
            <Box sx={{ display: "flex" }}>
              <Typography level="title-sm">{agent.name}</Typography>
              {agent.gender == "male" ? <Male /> : <Female />}
            </Box>
            <Typography level="body-sm" noWrap>
              {agent.status},{agent.age} years old
            </Typography>
          </ListItemContent>
        </AutocompleteOption>
      )}
      loading={isLoading}
      endDecorator={isLoading ? <CircularProgress size="sm" /> : null}
      size="sm"
      variant="outlined"
      placeholder="Search an agent"
      startDecorator={<SearchRounded color="primary" />}
      sx={{
        flexBasis: "500px",
        display: {
          xs: "none",
          sm: "flex",
        },
        boxShadow: "sm",
      }}
      inputValue={query}
      onInputChange={(_, value) => setQuery(value)}
      onChange={(_, value) => value && setDisplayedAgentId(value.id)}
    />
  );
}
