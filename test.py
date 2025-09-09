DEFAULT_HM = "hmmm..."

state = dict(hi="hello!", hm="broroooo")
state["hm"] = state["hm"] if "hm" in state else DEFAULT_HM
a = state["hm"]

print(a)
