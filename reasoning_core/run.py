# from my_task_test import AdditionTask, AdditionConfig
# from template import Config

# task = AdditionTask(config=AdditionConfig())

# easy = task.generate_example(level=0)
# hard = task.generate_example(level=5)


# print("QUESTION:", easy.prompt)
# print("ANSWER:", easy.answer)
# print("METADATA:", easy.metadata)

# print("QUESTION:", hard.prompt)
# print("ANSWER:", hard.answer)
# print("METADATA:", hard.metadata)


# from tasks.coding import CodeExecution, DiffPrediction, DiffPatching

# ### XXX: CodeExecution
# task = CodeExecution()
# ex = task.generate_example()

# print("=== CodeExecution ===")
# print(f"PROMPT:\n", ex.prompt)
# print(f"ANSWER:\n", ex.answer)

# ### XXX: DiffPrediction
# task = DiffPrediction()
# ex = task.generate_example()

# print("\n=== DiffPrediction ===")
# print("PROMPT:\n", ex.prompt)
# print("ANSWER:\n", ex.answer)


# ### XXX: DiffPatching
# task = DiffPatching()
# ex = task.generate_example()

# print("\n=== DiffPatching ===")
# print("PROMPT:\n", ex.prompt)
# print("ANSWER:\n", ex.answer)


from stateful_execution import StatefulExecution, StatefulConfig

task = StatefulExecution()

example = task.generate_example()

print("=== QUESTION ===")
print(example.prompt)

print("\n=== ANSWER ===")
print(example.answer)

print("\n=== METADATA ===")
print(example.metadata)

print("\n=== CoT (debug) ===")
print(example.metadata.cot)