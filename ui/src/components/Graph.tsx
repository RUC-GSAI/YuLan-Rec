import { DisplayedAgentDispatchContext } from "@/context/DisplayedAgentContext";
import { UiSettingsContext } from "@/context/UiSettingsContext";
import { useActiveAgents, useAgents } from "@/hooks/api/agent";
import { useRelationship } from "@/hooks/api/relationship";
import Agent from "@/types/api/agent";
import ReactEchartsCore from "echarts-for-react/lib/core";
import { GraphChart, GraphSeriesOption } from "echarts/charts";
import {
  GraphicComponent,
  GraphicComponentOption,
  TooltipComponent,
  TooltipComponentOption,
} from "echarts/components";
import * as echarts from "echarts/core";
import { ComposeOption, ECElementEvent } from "echarts/core";
import { SVGRenderer } from "echarts/renderers";
import { useContext, useEffect, useState } from "react";

function useGraph() {
  const {
    agents: initialAgents,
    error: agentsError,
    isLoading: agentsLoading,
  } = useAgents();
  const { agents: activeAgents } = useActiveAgents();
  const {
    relationship,
    error: relationshipsError,
    isLoading: relationshipsLoading,
  } = useRelationship();

  const error = agentsError || relationshipsError;
  const isLoading = agentsLoading || relationshipsLoading;

  if (!initialAgents || !relationship) {
    return {
      error,
      isLoading,
    };
  }

  let agents = initialAgents;

  if (activeAgents) {
    const agentsMap = new Map<number, Agent>();
    agents.concat(activeAgents).forEach((agent) => {
      agentsMap.set(agent.id, agent);
    });
    agents = Array.from(agentsMap.values());
  }

  return {
    graph: {
      nodes: agents,
      links: relationship,
    },
    error,
    isLoading,
  };
}

export default function Graph() {
  const uiSettings = useContext(UiSettingsContext);
  const setDisplayedAgentId = useContext(DisplayedAgentDispatchContext);

  const [onEvents] = useState({
    click: function (params: ECElementEvent) {
      if (params.dataType === "node") {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setDisplayedAgentId(parseInt((params.data as any).id));
      }
    },
  });

  echarts.use([
    // Main renderer
    SVGRenderer,

    // For loading animation and logo
    GraphicComponent,

    // Title
    GraphChart,
    TooltipComponent,
  ]);

  const { graph, error, isLoading } = useGraph();

  const loadingDuration = 3000;
  const [animationCompleted, setAnimationCompleted] = useState(false);
  const [graphInitialized, setGraphInitialized] = useState(false);
  useEffect(() => {
    setTimeout(() => {
      setAnimationCompleted(true);
    }, loadingDuration);
  }, []);

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (isLoading || !animationCompleted || !graph) {
    const option: ComposeOption<GraphicComponentOption> = {
      graphic: {
        elements: [
          {
            type: "text",
            left: "center",
            top: "center",
            style: {
              text: "RecAgent",
              fontSize: 80,
              fontWeight: "bold",
              lineDash: [0, 200],
              lineDashOffset: 0,
              fill: "transparent",
              stroke: "#000",
              lineWidth: 1,
            },
            keyframeAnimation: {
              duration: loadingDuration,
              loop: false,
              keyframes: [
                {
                  percent: 0.7,
                  style: {
                    fill: "transparent",
                    lineDashOffset: 200,
                    lineDash: [200, 0],
                  },
                },
                {
                  // Stop for a while.
                  percent: 0.8,
                  style: {
                    fill: "transparent",
                  },
                },
                {
                  percent: 1,
                  style: {
                    fill: "black",
                  },
                },
              ],
            },
          },
        ],
      },
    };
    return (
      <ReactEchartsCore
        echarts={echarts}
        notMerge={graphInitialized}
        option={option}
        style={{ width: "100%", height: "100%" }}
      />
    );
  }

  const option: ComposeOption<GraphSeriesOption | TooltipComponentOption> = {
    series: [
      {
        type: "graph",
        layout: "force",
        roam: true,
        force: {
          friction: 0,
        },
        label: {
          position: "right",
          formatter: "{b}",
        },
        nodes: graph.nodes.map((node) => {
          return {
            id: node.id.toString(),
            name: node.name,
            symbol:
              node.event.action_type === "idle"
                ? "image://api/" + node.idle_url
                : node.event.action_type === "watching"
                ? "image://api/" + node.watching_url
                : node.event.action_type === "chatting"
                ? "image://api/" + node.chatting_url
                : node.event.action_type === "posting"
                ? "image://api/" + node.posting_url
                : "",
            symbolSize:
              node.event.action_type === "idle"
                ? uiSettings.node_size
                : uiSettings.active_node_size,
            itemStyle: {
              shadowColor: "rgba(0, 0, 0, 0.3)",
              shadowBlur: 10,
              shadowOffsetX: 5,
              shadowOffsetY: 5,
            },
            category: node.event.action_type,
          };
        }),
        categories: [
          {
            name: "idle",
          },
          {
            name: "watch",
          },
          {
            name: "chat",
          },
        ],
        links: graph.links,
        emphasis: {
          focus: "adjacency",
          lineStyle: {
            width: 5,
          },
        },
      },
    ],
    tooltip: {
      position: (point) => {
        return [point[0] + 10, point[1] + 10];
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      formatter: (params: any) => {
        if (params.dataType === "node") {
          return `${params.data.name} is ${params.data.category}`;
        } else if (params.dataType === "edge") {
          const source = graph.nodes[params.data.source].name;
          const target = graph.nodes[params.data.target].name;
          return `${source} is ${params.data.name} of ${target}`;
        } else {
          return "";
        }
      },
    },
  };

  return (
    <ReactEchartsCore
      echarts={echarts}
      notMerge={!graphInitialized}
      option={option}
      style={{
        width: "100%",
        height: "100%",
        backgroundImage: `url(/background.png)`,
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundSize: "auto 100%",
      }}
      onEvents={onEvents}
      onChartReady={() => setGraphInitialized(true)}
    />
  );
}
