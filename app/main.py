from agents import search_agent, report_agent

query = "Future of quantum computing"

print("=== SEARCH AGENT ===")
research = search_agent(query)
print(research)

print("\n=== REPORT AGENT ===")
report = report_agent(query, research)
print(report)
