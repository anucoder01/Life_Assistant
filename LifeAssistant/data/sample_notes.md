# My Study Notes - Machine Learning

## Neural Networks - Chapter 5

### Backpropagation
Backpropagation computes gradients by applying the chain rule from output to input.
Key steps:
1. Forward pass: compute predictions
2. Compute loss (e.g., cross-entropy, MSE)
3. Backward pass: compute gradients layer by layer
4. Update weights using gradient descent

Formula: dL/dW = dL/dA * dA/dZ * dZ/dW

### Activation Functions
- ReLU: f(x) = max(0, x) — most common, fast, avoids vanishing gradient
- Sigmoid: f(x) = 1/(1+e^-x) — used for binary output
- Tanh: range [-1,1], zero-centered
- Softmax: for multi-class output, converts logits to probabilities

### Important: Vanishing Gradient Problem
Deep networks suffer when gradients become very small in early layers.
Solutions:
- Use ReLU instead of sigmoid/tanh
- Batch normalization
- Residual connections (ResNets)
- Careful weight initialization (He/Xavier)

---

## Python Project Notes - Data Pipeline

### Status: In Progress (deadline: Jan 17)
Completed modules:
- [x] data_ingestion.py — reads CSV/JSON from S3
- [x] data_cleaning.py — handles nulls, type casting
- [ ] data_transform.py — feature engineering (IN PROGRESS)
- [ ] data_loader.py — batch loading to Postgres

### Key issue to fix:
The chunked read in data_ingestion.py fails when files >2GB.
Need to implement streaming read with generator function.

### Dependencies:
- pandas 2.0, boto3, psycopg2, pydantic

---

## LeetCode Progress

### Trees & Graphs (Week 3)
Problems solved this week:
- Binary Tree Level Order Traversal ✓ (BFS approach)
- Lowest Common Ancestor ✓ (recursive)
- Word Ladder ✗ (need to revisit bidirectional BFS)
- Number of Islands ✓ (DFS/union-find)
- Course Schedule ✓ (topological sort)

### Pattern: DFS template for trees
```python
def dfs(node):
    if not node:
        return
    # process node
    dfs(node.left)
    dfs(node.right)
```

### Struggle areas:
- Graph problems with negative weights (Bellman-Ford)
- Bit manipulation tricks
- Need more practice: sliding window patterns

---

## Reading List
- [ ] "Designing Data-Intensive Applications" - Kleppmann (30% done)
- [ ] "Attention is All You Need" paper - Transformer architecture
- [ ] "The Pragmatic Programmer" - Hunt & Thomas
- [x] "Clean Code" - Robert Martin (completed Jan 5)
- [x] "Python Tricks" - Dan Bader (completed Dec 20)

---

## Personal Goals & Reflections

### January Goals
1. Finish ML course Chapters 5-8
2. Complete data pipeline project
3. Hit 75 LeetCode problems solved
4. Maintain workout streak > 14 days
5. Read 2 technical papers

### What's working
- Morning study sessions (9-11am) are most productive
- Pomodoro technique helping with focus
- GitHub streak motivation is real

### What needs improvement
- Procrastinating on difficult LeetCode problems
- Evening phone use disrupting sleep → affects morning energy
- Need better note-taking system for papers
