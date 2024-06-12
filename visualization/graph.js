const svg = d3.select("svg");

d3.json("../graphs/medium1k_mpnet_cosine_knn3_mean_kmeans100_x_lmao_100.json").then(
    function (graph) {
        const nodesData = graph.nodes;
        const linksData = graph.links;

        const width = window.innerWidth;
        const height = window.innerHeight;

        svg.attr("width", width).attr("height", height);

        const offsetX = width / 2;
        const offsetY = height / 2;

        const scaler = 4000;

        nodesData.forEach((node) => {
            node.pos[0] = node.pos[0] * scaler + offsetX;
            node.pos[1] = node.pos[1] * scaler + offsetY;
            node.originalPos = [node.pos[0], node.pos[1]];
        });

        function attractTowardsOriginalPosition(alpha) {
            const strength = 0.01; // attraction strength scaler
            nodesData.forEach((node) => {
                const targetX = node.originalPos[0];
                const targetY = node.originalPos[1];
                node.vx += (targetX - node.x) * alpha * strength;
                node.vy += (targetY - node.y) * alpha * strength;
            });
        }

        function repel(alpha) {
            const strength = 0.05; // repulsion strength scaler
            nodesData.forEach((node) => {
                nodesData.forEach((otherNode) => {
                    if (node !== otherNode) {
                        const dx = node.x - otherNode.x;
                        const dy = node.y - otherNode.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        const forceX = (dx / distance) * strength;
                        const forceY = (dy / distance) * strength;
                        node.vx += forceX * alpha;
                        node.vy += forceY * alpha;
                    }
                });
            });
        }

        const simulation = d3
            .forceSimulation(nodesData)
            .force("charge", d3.forceManyBody().strength(-10))
            .force("center", d3.forceCenter(offsetX, offsetY))
            .force(
                "link",
                d3
                    .forceLink(linksData)
                    .id((d) => d.id)
                    .strength((d) => d.weight * 0.2)
            )
            .force("collision", d3.forceCollide().radius(15))
            .force("attract", attractTowardsOriginalPosition)
            .force("repel", repel)
            .on("tick", ticked);

        const links = svg
            .selectAll(".link")
            .data(linksData)
            .enter()
            .append("line")
            .attr("class", "link")
            .style("stroke-width", (d) => Math.sqrt(d.weight * 5));

        const nodes = svg
            .selectAll(".node")
            .data(nodesData)
            .enter()
            .append("g")
            .attr("class", "node")
            .call(
                d3
                    .drag() // Enable dragging behavior
                    .on("start", dragStarted)
                    .on("drag", dragged)
                    .on("end", dragEnded)
            );

        nodes.append("circle")
            .attr("r", 10)
            .attr("fill", (d) => d.color)
            .on("mouseover", mouseover)
            .on("mouseout", mouseout);

        nodes.append("text")
            .attr("class", "clusterLabel")
            .text((d) => d.cluster)
            .attr("dx", -3)
            .attr("dy", 3)
            .style("font-size", "8px")
            .style("fill", "#fff");

        const labels = svg
            .selectAll(".nodeLabel")
            .data(nodesData)
            .enter()
            .append("text")
            .attr("class", "nodeLabel")
            .text((d) => d.title)
            .attr("dx", 12)
            .attr("dy", 4)
            .style("font-size", "7px");

        function ticked() {
            links
                .attr("x1", (d) => d.source.x)
                .attr("y1", (d) => d.source.y)
                .attr("x2", (d) => d.target.x)
                .attr("y2", (d) => d.target.y);

            nodes.attr("transform", (d) => `translate(${d.x},${d.y})`);

            labels.attr("x", (d) => d.x).attr("y", (d) => d.y);
        }

        function dragStarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragEnded(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        function mouseover(event, d) {
            d3.select(".tooltip")
                .transition()
                .duration(100)
                .style("opacity", .9);
            d3.select(".tooltip")
                .html(`
                    <div class="tooltip-content">
                        <h4>${d.title}</h4>
                        <p><strong>Tags:</strong> ${d.tags}</p>
                        <p><strong>Cluster Assignment:</strong> ${d.cluster}</p>
                    </div>
                `)
                .style("left", (event.pageX + 5) + "px")
                .style("top", (event.pageY - 28) + "px");

            d3.select(this).classed("highlight", true);

            // Highlight the connected edges
            links
                .filter(l => l.source.id === d.id || l.target.id === d.id)
                .classed("highlight", true);

            // Highlight the neighboring nodes
            nodes
                .filter(n => linksData.some(l => (l.source.id === d.id && l.target.id === n.id) || (l.target.id === d.id && l.source.id === n.id)))
                .classed("highlight-neighbor", true);
        }

        function mouseout(event, d) {
            d3.select(".tooltip")
                .transition()
                .duration(500)
                .style("opacity", 0);

            d3.select(this).classed("highlight", false);

            // Remove highlight from the connected edges
            links
                .filter(l => l.source.id === d.id || l.target.id === d.id)
                .classed("highlight", false);

            // Remove highlight from the neighboring nodes
            nodes
                .filter(n => linksData.some(l => (l.source.id === d.id && l.target.id === n.id) || (l.target.id === d.id && l.source.id === n.id)))
                .classed("highlight-neighbor", false);
        }
    }
);
