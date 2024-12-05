import "antd/dist/reset.css"; // 確保引入 Ant Design 的樣式
import * as d3 from "d3";
import React, { useEffect, useRef } from "react";

const GraphDemo = () => {
  const svgRef = useRef(null);

  useEffect(() => {
    // 定義圖表數據
    const data = {
      nodes: [
        { id: "Alice", label: "Alice" },
        { id: "Bob", label: "Bob" },
        { id: "Carol", label: "Carol" },
      ],
      links: [
        { source: "Alice", target: "Bob" },
        { source: "Bob", target: "Carol" },
      ],
    };

    const width = 600; // 圖表寬度
    const height = 400; // 圖表高度

    // 清除之前的 SVG 內容（避免重複渲染）
    d3.select(svgRef.current).selectAll("*").remove();

    // 創建 SVG
    const svg = d3
      .select(svgRef.current)
      .attr("width", width)
      .attr("height", height)
      .call(
        d3
          .zoom()
          .scaleExtent([0.5, 2])
          .on("zoom", (event) => {
            g.attr("transform", event.transform);
          })
      )
      .append("g");

    const g = svg.append("g");

    // 定義力導向圖
    const simulation = d3
      .forceSimulation(data.nodes)
      .force(
        "link",
        d3.forceLink(data.links).id((d) => d.id).distance(150)
      )
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(50));

    // 繪製連結
    const link = g
      .selectAll(".link")
      .data(data.links)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", "#87CEEB") // 天藍色邊
      .attr("stroke-width", 3);

    // 繪製節點
    const node = g
      .selectAll(".node")
      .data(data.nodes)
      .enter()
      .append("g")
      .attr("class", "node")
      .call(
        d3
          .drag()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // 繪製節點圓形，統一配色
    node
      .append("circle")
      .attr("r", 20)
      .attr("fill", "#FF6347") // 番茄紅色
      .attr("stroke-width", 3);

    // 添加節點標籤
    node
      .append("text")
      .attr("dy", 5)
      .attr("text-anchor", "middle")
      .text((d) => d.label)
      .attr("font-size", 12)
      .attr("fill", "#FFFFFF"); // 白色字體

    // 更新節點和連結的位置
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    // 清理函數
    return () => {
      simulation.stop();
    };
  }, []);

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        width: "100%",
        height: "100%",
        margin: "20px auto",
      }}
    >
      <svg
        ref={svgRef}
        style={{
          width: "60%",
          height: "auto",
          border: "1px solid #333",
        }}
      ></svg>
    </div>
  );
};

export default GraphDemo;
