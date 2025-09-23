import pygraphviz as pgv
from IPython.display import Image, Markdown, display

# https://garba.org/posts/2022/graph/#multigraph

def visualise(graph,hide=False,clusters=None,circular=False):
    layout = "circo" if circular else "dot"
    oneblock = True if circular else False
    G = pgv.AGraph(strict=False, directed=graph.is_directed(),
                    rankdir="LR",newrank="True",layout=layout,oneblock=oneblock)

    targetGraph = {}

    if clusters is not None:
        for (label,cluster) in clusters:
            name = "cluster_%s" % label.lower().replace(" ","_")
            subgraph = G.add_subgraph(name=name,label=label,labelloc="b",rank="same")
            for node in cluster:
                targetGraph[node] = subgraph
    else:
        for n in graph.nodes():
            targetGraph[n] = G

    for (n,data) in graph.nodes(data=True):
        label = "%s " % n
        for attribute,value in data.items():
            if not hide:
                label +=  "\n%s = " % attribute
            label += "\n%s," % value
        if len(label) > 1:
            label = label[0:-1]
        targetGraph[n].add_node(n,label=label)
    for (u,v,data) in graph.edges(data=True):

        dir = 'forward' if graph.is_directed() else 'none'
        label = ""
        for attribute,value in data.items():
            if not hide:
                label +=  " %s = " % attribute
            label += "%s," % value
        if len(label) > 1:
            label = label[0:-1]
        G.add_edge(u,v,dir=dir,label=label)
        G.layout(prog="dot")
    display(Image(G.draw(format='png')))
    return
