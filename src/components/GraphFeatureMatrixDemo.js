import "antd/dist/reset.css"; // 確保引入 Ant Design 的樣式
import * as d3 from "d3";
import React, { useEffect, useRef } from "react";

const GraphFeatureMatrixDemo = () => {
  const svgRef = useRef(null);

  useEffect(() => {
    const data = {
      nodes: [
        { id: "Alice", label: "Alice", age: 35, exercise: 1 },
        { id: "Bob", label: "Bob", age: 50, exercise: 0 },
        { id: "Carol", label: "Carol", age: 22, exercise: 1 },
      ],
      links: [
        { source: "Alice", target: "Bob" },
        { source: "Bob", target: "Carol" },
      ],
    };

    const width = 600; // 基準寬度
    const height = 400; // 基準高度

    // 清除之前的 SVG 內容（避免重複渲染）
    d3.select(svgRef.current).selectAll("*").remove();

    // 設定 SVG 的 viewBox，以達成自適應
    const svg = d3
      .select(svgRef.current)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .call(
        d3
          .zoom()
          .scaleExtent([0.5, 2])
          .on("zoom", (event) => {
            g.attr("transform", event.transform);
          })
      );

    const g = svg.append("g");

    // 定義力導向圖
    const simulation = d3
      .forceSimulation(data.nodes)
      .force("link", d3.forceLink(data.links).id((d) => d.id).distance(150))
      .force("charge", d3.forceManyBody().strength(-400))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) => d.age / 2 + 20));

    // 繪製連結
    const link = g
      .selectAll(".link")
      .data(data.links)
      .enter()
      .append("line")
      .attr("class", "link")
      .attr("stroke", "#aaa")
      .attr("stroke-width", 2);

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

    // 根據年齡、運動習慣設定節點外觀
    node
      .append("circle")
      .attr("r", (d) => d.age)
      .attr("fill", (d) => (d.exercise ? "#69b3a2" : "#cccccc"))
      .attr("stroke", "#333")
      .attr("stroke-width", 2);

    // 添加節點標籤
    node
      .append("text")
      .attr("dy", 5)
      .attr("text-anchor", "middle")
      .text((d) => d.label)
      .attr("font-size", 12)
      .attr("fill", "#000");

    // 添加提示
    node
      .append("title")
      .text(
        (d) =>
          `Name: ${d.label}, Age: ${d.age}, Exercise: ${
            d.exercise ? "Yes" : "No"
          }`
      );

    // 更新位置
    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);

      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

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
        maxWidth: "600px",  // 可自行調整最大寬度
        margin: "20px auto",
      }}
    >
      <svg
        ref={svgRef}
        style={{
          width: "100%",  // 設定SVG寬度100%讓其自動縮放
          height: "auto",
          border: "1px solid #333",
        }}
      ></svg>
    </div>
  );
};

export default GraphFeatureMatrixDemo;
