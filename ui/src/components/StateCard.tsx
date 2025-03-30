import { useSystemState } from "@/hooks/api/state";
import { ExpandLess, ExpandMore } from "@mui/icons-material";
import {
  Card,
  CardContent,
  IconButton,
  Sheet,
  Tab,
  TabList,
  TabPanel,
  Tabs,
  Tooltip,
  Typography,
  tabClasses,
} from "@mui/joy";
import { SxProps } from "@mui/joy/styles/types";
import { useState } from "react";

export default function StateCard({ sx }: { sx?: SxProps }) {
  const { state } = useSystemState();
  const recommenderState = state?.recommender;
  const socialState = state?.social;

  const [contentVisible, setContentVisible] = useState(true);

  return (
    <Card variant="outlined" sx={{ ...sx }}>
      <Sheet
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Typography level="title-lg">State Information</Typography>
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
        <CardContent>
          <Tabs>
            <TabList
              tabFlex={1}
              disableUnderline
              sx={{
                p: 0.5,
                gap: 0.5,
                borderRadius: "xl",
                bgcolor: "background.level1",
                [`& .${tabClasses.root}[aria-selected="true"]`]: {
                  boxShadow: "sm",
                  bgcolor: "background.surface",
                },
              }}
            >
              <Tab disableIndicator>Recommender</Tab>
              <Tab disableIndicator>Social</Tab>
            </TabList>
            <TabPanel value={0} sx={{ p: 0 }}>
              <Sheet
                variant="soft"
                sx={{
                  borderRadius: "md",
                  p: 1.5,
                  mt: 1.5,
                }}
              >
                <Sheet
                  variant="soft"
                  sx={{
                    display: "flex",
                    gap: 2,
                    "& > div": { flex: 1 },
                  }}
                >
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Total Users
                    </Typography>
                    <Typography fontWeight="lg" noWrap>
                      {recommenderState?.tot_user_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Total Movies
                    </Typography>
                    <Typography fontWeight="lg">
                      {recommenderState?.tot_item_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg">
                      Algorithm
                    </Typography>
                    <Typography fontWeight="lg">
                      {recommenderState?.rec_model}
                    </Typography>
                  </div>
                </Sheet>
                <Sheet
                  variant="soft"
                  sx={{
                    display: "flex",
                    gap: 2,
                    "& > div": { flex: 1 },
                  }}
                >
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Interactions
                    </Typography>
                    <Typography fontWeight="lg" noWrap>
                      {recommenderState?.inter_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Current Users
                    </Typography>
                    <Typography fontWeight="lg">
                      {recommenderState?.cur_user_num}
                    </Typography>
                  </div>
                </Sheet>
                <Sheet
                  variant="soft"
                  sx={{
                    gap: 2,
                    "& > div": { flex: 1 },
                  }}
                >
                  <Typography level="body-xs" fontWeight="lg" noWrap>
                    Most Popular Movie
                  </Typography>
                  {recommenderState?.pop_items?.map((item, index) => (
                    <Sheet
                      key={index}
                      variant="soft"
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography fontWeight="lg">#{index + 1}</Typography>
                      <Typography fontWeight="lg" fontStyle="italic" noWrap>
                        {item.replace(/^<+|>+$/g, "")}
                      </Typography>
                    </Sheet>
                  ))}
                </Sheet>
              </Sheet>
            </TabPanel>
            <TabPanel value={1} sx={{ p: 0 }}>
              <Sheet
                variant="soft"
                sx={{
                  borderRadius: "md",
                  p: 1.5,
                  mt: 1.5,
                }}
              >
                <Sheet
                  variant="soft"
                  sx={{
                    display: "flex",
                    gap: 2,
                    "& > div": { flex: 1 },
                  }}
                >
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Total Users
                    </Typography>
                    <Typography fontWeight="lg" noWrap>
                      {socialState?.tot_user_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Total Links
                    </Typography>
                    <Typography fontWeight="lg">
                      {socialState?.tot_link_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Network Density
                    </Typography>
                    <Typography fontWeight="lg">
                      {socialState?.network_density}
                    </Typography>
                  </div>
                </Sheet>
                <Sheet
                  variant="soft"
                  sx={{
                    display: "flex",
                    gap: 2,
                    "& > div": { flex: 1 },
                  }}
                >
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Total Chats
                    </Typography>
                    <Typography fontWeight="lg" noWrap>
                      {socialState?.chat_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Total Posts
                    </Typography>
                    <Typography fontWeight="lg">
                      {socialState?.post_num}
                    </Typography>
                  </div>
                  <div>
                    <Typography level="body-xs" fontWeight="lg" noWrap>
                      Current Chats
                    </Typography>
                    <Typography fontWeight="lg">
                      {socialState?.chat_num}
                    </Typography>
                  </div>
                </Sheet>
                <Sheet
                  variant="soft"
                  sx={{
                    gap: 2,
                    "& > div": { flex: 1 },
                  }}
                >
                  <Typography level="body-xs" fontWeight="lg" noWrap>
                    Most Talked Movie
                  </Typography>
                  {socialState?.pop_items?.map((item, index) => (
                    <Sheet
                      key={index}
                      variant="soft"
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography fontWeight="lg">#{index + 1}</Typography>
                      <Typography fontWeight="lg" fontStyle="italic" noWrap>
                        {item.replace(/^<+|>+$/g, "")}
                      </Typography>
                    </Sheet>
                  ))}
                </Sheet>
              </Sheet>
            </TabPanel>
          </Tabs>
        </CardContent>
      )}
    </Card>
  );
}
