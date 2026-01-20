# 🎓 Academic Portfolio — Master's in Artificial Intelligence
 
**Anna Monso Rodriguez**
Master in Artificial Intelligence | Illinois Institute of Technology
[GitHub](https://github.com/annamonso) | [LinkedIn](https://linkedin.com/in/your-profile)
 
---
 
## 📌 Overview
 
This repository showcases a comprehensive collection of graduate-level coursework in **Artificial Intelligence**, **Machine Learning**, **Deep Learning**, and **Social Network Analysis**. Each project demonstrates hands-on implementation of fundamental and advanced AI concepts, from classical search algorithms to modern deep learning architectures.
 
**Key Focus Areas:**
- Algorithm design and optimization (search, game theory, constraint satisfaction)
- Deep learning fundamentals (backpropagation, RNNs, CNNs, autograd)
- Machine learning theory and practice (regression, regularization, kernel methods)
- Network science and graph analytics (bipartite graphs, community detection, centrality)
 
---
 
## 🗂️ Repository Structure
 
```
CourseWork_projects/
├── CS489 - Artificial Intelligence I/
│   ├── HW1/  Tic-Tac-Toe AI with Minimax & Alpha-Beta Pruning
│   └── HW2/  Constraint Satisfaction Problem (CSP) Solver
│
├── CS577 - Deep Learning/
│   ├── HW1/  Neural Network Training & Backpropagation from Scratch
│   ├── HW2/  Custom Autograd Functions & Gradient Computation
│   └── HW3/  im2col Implementation & 2-Layer RNN for Sequence Modeling
│
├── CS579 - Online Social Network Analysis/
│   └── ASS2/  Network Graph Analysis & Community Detection
│
└── CS584 - Machine Learning/
    ├── HW0/  Python & ML Fundamentals
    ├── HW1/  Polynomial Regression & Model Selection
    ├── HW2/  Multivariate Regression & Feature Engineering
    └── HW3/  Ensemble Learning & Kernel Methods
```
 
---
 
## 🧠 CS489 — Artificial Intelligence I
 
### **HW1: Tic-Tac-Toe Agent (Minimax & Alpha-Beta Pruning)**
- Implemented game-playing AI using **Minimax algorithm** with **Alpha-Beta pruning optimization**
- Designed evaluation functions and terminal state detection
- Built interactive command-line game interface
- **Skills:** Game tree search, adversarial search, algorithm optimization
 
### **HW2: Constraint Satisfaction Problem Solver**
- Developed **backtracking algorithm** for CSP with constraint propagation
- Applied to route planning problem across multiple zones
- Implemented efficient data loading and validation from CSV files
- **Skills:** Search algorithms, constraint reasoning, algorithmic problem-solving
 
**Technologies:** Python, algorithmic design, complexity analysis
 
---
 
## 🤖 CS577 — Deep Learning
 
### **HW1: Neural Network Training from Scratch**
- Built 2-layer neural network with **ReLU activations** using NumPy
- Manually derived and implemented **backpropagation** for gradient computation
- Trained model using **gradient descent** with learning rate tuning
- Analyzed convergence behavior and loss landscapes
- **Skills:** Gradient calculus, numerical optimization, neural network fundamentals
 
### **HW2: Custom Autograd Functions**
- Extended PyTorch's autograd system with custom `max()` and `min()` functions
- Implemented forward and backward passes for computational graph operations
- Traced gradient flow through complex computational graphs
- Debugged numerical stability issues in gradient computation
- **Skills:** Automatic differentiation, computational graphs, PyTorch internals
 
### **HW3: Convolutional Operations & RNNs**
- **Part 1:** Extended `im2col` function for flexible patch extraction in CNNs
- **Part 2:** Implemented **2-layer RNN** to predict top-2 elements in sequences
- Analyzed vanishing gradient problems in recurrent architectures
- **Skills:** CNNs, RNNs, sequence modeling, tensor operations
 
**Technologies:** Python, NumPy, PyTorch, Jupyter Notebooks, deep learning theory
 
---
 
## 🌐 CS579 — Online Social Network Analysis
 
### **Assignment 2: Graph Analysis & Community Detection**
 
**Dataset 1: Chicago Community Networks**
- Analyzed 77 Chicago community areas as a graph network
- Computed degree distributions and identified key network properties
- Visualized geographic network topology with NetworkX
- Found Edison Park (degree 1) and Auburn Gresham (degree 8) as extreme connectivity cases
 
**Dataset 2: Student Attribute Network**
- Built **bipartite graph** connecting students to attributes (hobbies, languages, clubs)
- Created **unimodal projection** to analyze student similarity networks
- Performed extensive data cleaning and normalization (200+ unique hobby entries clustered)
- Analyzed network density and identified shared universal traits
 
**Key Findings:**
- Demonstrated difference between bipartite and unimodal graph representations
- Identified limitations of unimodal graphs when attributes are too common
- Proposed filtering strategies for more granular community insights
 
**Technologies:** Python, NetworkX, pandas, matplotlib, graph theory, data cleaning
 
---
 
## 📊 CS584 — Machine Learning
 
### **HW1: Polynomial Regression & Model Selection**
 
**Single-Variable Regression:**
- Compared **OLS (Normal Equations)** vs **Gradient Descent** implementations
- Performed polynomial degree selection (2–10) using 10-fold cross-validation
- Analyzed bias-variance tradeoff through learning curves
- Evaluated **Ridge** and **Lasso** regularization for numerical stability
 
**Key Results (4 datasets):**
- Dataset 1 (linear): Degree 6 optimal, OLS best balance (RMSE 2.06, R² 0.97)
- Dataset 2 (nonlinear): Degree 10 optimal, polynomial outperformed linear (RMSE 0.05 vs 0.22)
- Dataset 3 (Gaussian): Degree 10 captured unimodal structure effectively
- Dataset 4 (multimodal): Polynomial model essential (RMSE 0.46 vs 0.99 linear)
 
**Skills:** Regression analysis, cross-validation, regularization, model selection, bias-variance analysis
 
---
 
### **HW2: Multivariate Regression & Feature Engineering**
 
- Generated polynomial features up to degree 3 and applied variance/correlation filtering
- Compared **Normal Equation** vs **Gradient Descent** solvers (time/accuracy tradeoffs)
- Implemented **Ridge regression** with hyperparameter tuning via cross-validation
- Applied **Huber regression** for outlier robustness
- Analyzed condition numbers and numerical stability of design matrices
 
**Key Insights:**
- Extra features improved accuracy when paired with Normal Equations
- Ridge regularization reduced condition number by orders of magnitude
- Huber regression showed minimal improvement when outliers were moderate
 
---
 
### **HW3: Kernel Methods & Real-World Application**
 
**Kernel Ridge Regression:**
- Implemented **RBF kernel** with dual formulation
- Performed 2D grid search over λ and γ hyperparameters
- Compared **primal ridge** (explicit features) vs **kernel ridge** (implicit features)
 
**Real-Data Study (Temperature Forecasting):**
- Analyzed meteorological dataset with 7,500+ observations
- Preprocessed features: scaling, missing value imputation
- Compared OLS, Ridge, Huber, and Kernel Ridge regression
- Generated learning curves and repeated cross-validation for variance estimation
 
**Key Findings:**
- Kernel ridge achieved better accuracy but 100× higher memory usage
- Ridge and OLS performed similarly (RMSE ~1.50), indicating well-conditioned data
- Demonstrated practical constraints of kernel methods on large datasets
 
**Technologies:** Python, NumPy, pandas, scikit-learn, matplotlib, GridSearchCV
 
---
 
## 🛠️ Technical Skills Demonstrated
 
### **Programming & Tools**
- **Languages:** Python (NumPy, pandas, scikit-learn, PyTorch)
- **Visualization:** matplotlib, NetworkX
- **Development:** Jupyter Notebooks, Git version control
 
### **Machine Learning**
- Supervised learning (regression, classification)
- Model selection & hyperparameter tuning
- Regularization techniques (Ridge, Lasso, kernel methods)
- Bias-variance analysis & learning curves
- Cross-validation & performance metrics
 
### **Deep Learning**
- Neural network architectures (MLPs, CNNs, RNNs)
- Backpropagation & gradient computation
- Autograd systems & computational graphs
- PyTorch framework internals
 
### **Algorithms & Theory**
- Search algorithms (Minimax, backtracking, CSP)
- Graph algorithms & network analysis
- Optimization (gradient descent, closed-form solutions)
- Complexity analysis & algorithm design
 
### **Data Science**
- Data cleaning & preprocessing
- Feature engineering & dimensionality reduction
- Statistical analysis & visualization
- Real-world dataset handling
 
---
 
## 🎯 Project Highlights for Recruiters
 
### **Strong Foundations**
- Implemented core algorithms from scratch (neural networks, backpropagation, game AI)
- Deep understanding of mathematical foundations (linear algebra, calculus, probability)
- Rigorous experimental methodology with proper validation
 
### **Practical Skills**
- End-to-end ML pipeline: data cleaning → feature engineering → model selection → evaluation
- Handled real-world datasets with missing values, outliers, and high dimensionality
- Balanced theoretical understanding with practical implementation
 
### **Problem-Solving Ability**
- Debugged numerical stability issues in gradient computation
- Optimized algorithms for efficiency (Alpha-Beta pruning, im2col operations)
- Analyzed tradeoffs between model complexity, accuracy, and computational cost
 
### **Communication**
- Comprehensive documentation and analysis in all assignments
- Clear visualizations of results (learning curves, heatmaps, network graphs)
- Ability to explain technical concepts and derive insights from experiments
 
---
 
## 📈 Learning Outcomes
 
Through these projects, I developed:
 
1. **Theoretical depth** in AI/ML fundamentals (optimization, probability, linear algebra)
2. **Implementation skills** to build algorithms from mathematical principles
3. **Experimental rigor** through cross-validation, ablation studies, and comparative analysis
4. **Engineering mindset** balancing accuracy, efficiency, and scalability
5. **Research skills** to interpret results, identify limitations, and propose improvements
 
---
 
## 📄 License
 
MIT License © 2025 Anna Monso Rodriguez
 
---
 
## 📫 Contact
 
**Anna Monso Rodriguez**
Master in Artificial Intelligence Student
Illinois Institute of Technology
 
- **GitHub:** [github.com/annamonso](https://github.com/annamonso)
- **LinkedIn:** [linkedin.com/in/your-profile](#)
- **Email:** your.email@example.com
 
---
 
*This portfolio represents academic work completed as part of the Master's in Artificial Intelligence program at Illinois Institute of Technology. All implementations follow academic integrity guidelines and are meant to demonstrate technical competency for professional opportunities.*