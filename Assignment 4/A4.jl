### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ dc004086-db6e-4813-841e-d427520402f7
begin
	using CSV, DataFrames, StatsPlots, PlutoUI, Random, Statistics
	using LinearAlgebra: dot, norm, norm1, norm2, I
	using Distributions: Distributions, Uniform, TDist, cdf, Normal
	using MultivariateStats: MultivariateStats, PCA
	using StatsBase: StatsBase
	
end

# ╔═╡ 75441ce6-2137-4fcf-bba2-6ed67b9acb59
begin
	_check_complete(complete) = complete ? "✅" : "❌"
	
	md"""
	# Setup

	this section loads and installs all the packages. You should be setup already from assignment 1, but if not please read and follow the `instructions.md` for further details.
	"""
end

# ╔═╡ c1b9d7a3-5605-4478-b1a9-a9553ad38ed7
PlutoUI.TableOfContents(title="A4 Outline")

# ╔═╡ c9797979-7d16-4ba4-8f82-c7ec33f6e517
plotly() # In this notebook we use the plotly backend for Plots.

# ╔═╡ 693a3933-c1c2-4249-8c03-f5151267222f
md"""
### !!!IMPORTANT!!!

Insert your details below. You should see a green checkmark.
"""

# ╔═╡ def97306-1703-42bc-bc09-da623c545e87
student = (name="Nasif Hossain", email="nhossain@ualberta.ca", ccid="nhossain", idnumber=1545143)

# ╔═╡ bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
let
	def_student = (name="NAME as in eclass", email="UofA Email", ccid="CCID", idnumber=0)
	if length(keys(def_student) ∩ keys(student)) != length(keys(def_student))
		md"You don't have all the right entries! Make sure you have `name`, `email`, `ccid`, `idnumber`. ❌"
	elseif any(getfield(def_student, k) == getfield(student, k) for k in keys(def_student))
		md"You haven't filled in all your details! ❌"
	elseif !all(typeof(getfield(def_student, k)) === typeof(getfield(student, k)) for k in keys(def_student))
		md"Your types seem to be off: `name::String`, `email::String`, `ccid::String`, `idnumber::Int`"
	else
		md"Welcome $(student.name)! ✅"
	end
end

# ╔═╡ 14c30b50-157f-40bf-b0a6-0232aa2705c6
md"""
Important Note: You should only write code in the cells that has: """


# ╔═╡ 4a196e82-fe94-49fe-ab87-d57e2d7fbd34
#### BEGIN SOLUTION


#### END SOLUTION

# ╔═╡ 7247e88f-e9c3-4561-ae67-71eb23ad066d
function check_elements(Φ_true::Matrix{Float64}, Φ::Matrix{Float64})
	all_exist = true
	for n in eachindex(Φ_true), m in eachindex(Φ_true[n])
		if Φ_true[n][m] ∉ Φ || Φ[n][m] ∉ Φ_true
			has_sim = false
			for i in eachindex(Φ_true), j in eachindex(Φ_true[i])
				if Φ_true[i][j] ≈ Φ[n][m]
					has_sim = true
				end 
			end 
			true_has_sim = false
			for i in eachindex(Φ), j in eachindex(Φ[i])
				if Φ[i][j] ≈ Φ_true[n][m]
					true_has_sim = true
				end 
			end 
			if has_sim == false || true_has_sim == false
				all_exist = false
			end
		end
	end
	all_exist
end 


# ╔═╡ a7aecd21-13f2-4cf2-8239-3a3d708602c9
md"""
# Q1: Multi-variate Binary Classification

So far, we have only considered algorithms for regression where the target is continuous. We will now explore implementations of algorithms for multi-variate binary classification.

As before, we have broken our ML systems into smaller pieces. This will allow us to more easily take advantage of code we've already written, and will be more useful as we expand the number of algorithms we consider. We make several assumptions to simplify the code, but the general type hierarchy can be used much more broadly.

We split each system into:
- Model
- Gradient descent procedure
- Loss Function
- Optimization Strategy
"""

# ╔═╡ e3c3e903-e2c2-4de0-a3b2-56a27709e8c3
md"""
## Baselines
The only baseline we will use in this assignment is a random classifier. 
"""

# ╔═╡ a35944ae-7bbb-4f3c-bbe0-d2bbb57cd70b
md"""
### RandomModel
"""

# ╔═╡ 7b513344-1cad-4eef-9faf-e77ba176323e
md"""
# Models

"""

# ╔═╡ 4f4029a2-c590-4bd3-a0db-d2380d4b4620
md"""
## The model interface

- `AbstractModel`: This is an abstract type which is used to derive all the model types in this assignment
- `predict`: This takes a matrix of samples and returns the prediction doing the proper data transforms.
- `get_features`: This transforms the features according to the non-linear transform of the model (which is the identity for linear).
- `get_linear_model`: All models are based on a linear model with transformed features, and thus have a linear model.
- `copy`: This returns a new copy of the model.
"""

# ╔═╡ dcfecc35-f25c-4856-8429-5c31d94d0a42
"""
	AbstractModel

Used as the root for all models in this notebook. We provide a helper `predict` function for `AbstractVectors` which transposes the features to a row vector. We also provide a default `update_transform!` which does nothing.
"""
abstract type AbstractModel end

# ╔═╡ d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
predict(alm::AbstractModel, x::AbstractVector) = predict(alm, x')[1];

# ╔═╡ 7cd46d84-a74c-44dc-8339-68010924bc39
update_transform!(AbstractModel, args...) = nothing;

# ╔═╡ fd4df2c6-fbfe-4089-a487-e3965a463ef3
md"
#### Linear Model

As before, we define a linear model that inputs a vector $\mathbf{x}$ and outputs prediction $\mathbf{x}^\top\mathbf{w}$. We can also input the whole data matrix $$\mathbf{X}$$ and output a prediction for each row using $$\mathbf{X} \mathbf{w}$$. 
We exploit this in `predict`, to return predictions for the data matrix $$\mathbf{X}$$ of size `(samples, features)`.

We define `get_features`, which we will need for polynomial regression. For logistic regression, the default is to return the inputs themselves. In polynomial logistic regression, we will replace this function with one that returns polynomial features.

"

# ╔═╡ 2d43a3ba-2a2c-4114-882f-5834d42e302a
begin
	struct LinearModel <: AbstractModel
		w::Matrix{Float64} # Aliased to Array{Float64, 2}
	end
	
	LinearModel(in, out=1) = 
		LinearModel(zeros(in, out)) # feature size × output size
	
	Base.copy(lm::LinearModel) = LinearModel(copy(lm.w))
	predict(lm::LinearModel, X::AbstractMatrix) = X * lm.w
	get_features(m::LinearModel, x) = x

end;

# ╔═╡ 9c0491db-ad27-46fc-969c-c42a14cdadeb
md"
#### Logistic Regression Model
Logistic regression is very similar to linear regression. But, unlike linear regression where $y$ is a continuous variable, the targets are binary $y \in \{0,1\}$ for logistic regression. Instead, we directly learn $p(y | \mathbf{x})$ with logistic regression, and then return the prediction $\hat{y}$ that has higher probability.
To limit the range of the learned probabilities within $[0,1]$, we use a *sigmoid* transformation. 
```math
p(y=1 | \mathbf{x})=σ(w_0+w_1x_1+w_2x_2+..+w_𝑛x_𝑛)
```
where $w_0$ represents the bias term. To take the bias term into account, we need to add a column of $1$s to the input matrix $$\mathbf{X}$$, resulting in the new dimension `(samples, features+1)`.

In this part you need to implement the sigmoid function. The sigmoid function takes in a scalar $z$ and is equal to $\text{sigmoid}(z) = 1/(1+ \exp(-z))$. To implement it, we can more generally assume it inputs a vector $\mathbf{z} = (z_1, z_2, \ldots, z_n)$ and outputs the sigmoid function applied to each element of this $\mathbf{z}$ (elementwise). Namely, $\text{sigmoid}(\mathbf{z}) = (1/(1+ \exp(-z_1)), 1/(1+ \exp(-z_2)), \ldots, 1/(1+ \exp(-z_n)))$. You can either implement this with a for loop, or use the fact that we can do elementwise operations on vectors. The `exp` function can be applied elementwise using `exp.(-z)`. And as before, elementwise addition is .+ and elementwise division is ./. If you find bugs using elementwise operations, then start with the straightforward for loop approach, get that working, and then experiment with elementwise operations.  
"

# ╔═╡ 0005ee8a-fcef-4c4e-8aa5-6140b06ed4ef
begin
	struct LogisticRegressor <: AbstractModel
		model::LinearModel
		γ::Float64 # the probabilty threshold on the output class confidence
		is_poly::Bool
	end
	
	LogisticRegressor(in, out=1; γ=0.5, is_poly=false) = if is_poly
		in = in - 1
		LogisticRegressor(LinearModel(in+1, out), γ, is_poly) # (feture size + 1 for bias term)  × output size
	else 
		LogisticRegressor(LinearModel(in+1, out), γ, is_poly) # (feture size + 1 for bias term)  × output size
	end
	Base.copy(lr::LogisticRegressor) = LogisticRegressor(copy(lr.model),lr.γ,lr.is_poly)
	get_linear_model(lr::LogisticRegressor) = lr.model
end;

# ╔═╡ 51599971-4638-4787-9c13-75afa0d34285
# Add a column of 1 to X to count for the bias term. Start with an "else" statement. 
function get_features(m::LogisticRegressor, X::AbstractMatrix)
	d = size(X, 2)
	_X = ones(size(X,1), d+1)
	_X[:, 1:d] = X
	X = _X
end;

# ╔═╡ 8847c374-c9f4-470d-80be-2946f2af9661
function sigmoid(z)
	z
	#### BEGIN SOLUTION
	for i in 1:size(z,1)
		for j in 1:size(z,2)
			z[i,j] = 1/(1 + exp(-z[i,j]))
		end 
	end
	return z
	#### END SOLUTION
end;

# ╔═╡ 8745fec1-47c8-428b-9ea4-1e6828618830
begin
	__check_logit_reg = let
		rng = Random.MersenneTwister(1)
		_X = rand(rng, 3, 3)
		X = sigmoid(_X)
		X_true = [0.5587358993498943 0.5019773105398053 0.7215004060928302; 0.5857727098994119 0.6197795961579493 0.7310398330188039; 0.5775458635048137 0.5525472988567002 0.562585578409889]
		all(X .≈ X_true)
		end
	HTML("<h2 id=dist> Q1: Logistic Regression $(_check_complete(__check_logit_reg))")
end


# ╔═╡ ded749bf-b9fa-4e2b-b15f-0693d820a9c3
md"""
Now, we will implement Polynomial Model which uses the linear model for learning on top of non-linear features. We apply a polynomial transformation to our data, like the previous assignment, but now do a higher degree of $p=3$. 

Recall from the last assignment to obtain a polynomial transformation to our data to create new polynomial features. For $d$ inputs with a polynomial of size $p$, the number of features is $m = {d+p \choose p}$, giving polynomial function 

```math
f(\mathbf{x})=\sum_{j=1}^{m} w_j \phi_j (\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^\top\mathbf{w}
```
We simply apply this transformation to every data point $\mathbf{x}_i$ to get the new dataset $\{(\boldsymbol{\phi}(\mathbf{x}_i), y_i)\}$.

Implement the polynomial feature transformation by constructing $\Phi$ with $p = 3$ degrees in the function ```get_features```.

"""

# ╔═╡ aa1dfa87-a844-4f82-9a30-008f15f88112
begin
	struct Polynomial3Model <: AbstractModel 
		model::LogisticRegressor
		ignore_first::Bool
	end
	
	Polynomial3Model(in, out=1; ignore_first=false) =
		Polynomial3Model(LogisticRegressor(1 + in + Int(in*(in+1)/2) + Int(floor((in*(in+1)/2)*(in+2)/3.0)), out, is_poly=true), ignore_first)

	Base.copy(lm::Polynomial3Model) = Polynomial3Model(copy(lm.model), lm.ignore_first)
	get_linear_model(lm::Polynomial3Model) = lm.model.model
	
end

# ╔═╡ 0ba5f9c8-5677-40e9-811b-25546e0df207
function get_features(pm::Polynomial3Model, _X::AbstractMatrix)
	# If _X already has a bias remove it.
	X = if pm.ignore_first
		_X[:, 2:end]
	else
		_X
	end
	
	m = size(X, 2)
	N = size(X, 1)
	num_features = 1 + # Bias bit
				   m + # p = 1
				   Int(m*(m+1)/2) + # combinations (i.e. x_i*x_j)
				   Int(floor(Int(m*(m+1)/2) * (m+2)/3))  # combinations (i.e. x_i*x_j*x_k)
	
	Φ = zeros(N, num_features)
	# Construct Φ
	#### BEGIN SOLUTION
	rows, cols = size(X)
	for i in  1:rows
		count = 1
		Φ[i, count] = 1.0
		count += 1
		for j in 1:cols
			Φ[i, count] = X[i,j]
			count += 1
			for k in j+1:cols
				Φ[i, count] = X[i,j] * X[i,k]
				count += 1
			end
			Φ[i, count] = X[i,j] * X[i,j]
			count = count + 1
		end

		
		for j in 1:cols
			for k in j+1:cols
				Φ[i, count] = X[i,j] * X[i,j] * X[i,k]
				count = count + 1
				Φ[i, count] = X[i,j] * X[i,k]* X[i,k]
				count = count + 1
			end
			Φ[i, count] = X[i,j] * X[i,j] * X[i,j]
			count = count + 1
		end		
				
	end

	#### END SOLUTION
	
	Φ
end;

# ╔═╡ 50cb6e7f-3341-47b8-b720-d069f03f1be2
function predict(lr::LogisticRegressor, X::AbstractMatrix)
	if lr.is_poly
		Ŷ = sigmoid(predict(lr.model, X))
	else
		Ŷ = sigmoid(predict(lr.model, get_features(lr, X))) 
	end
	pred = zeros(size(Ŷ))
	for i in 1:length(Ŷ)
		if Ŷ[i] >= lr.γ
			pred[i] = 1.0
		else
			pred[i] = 0.0
		end
	end
	pred
end;

# ╔═╡ c59cf592-a893-4ffa-b247-51d94c7cdb1a
begin
		
	__check_Poly2_logit_reg = let
		pm = Polynomial3Model(2, 1)
		rng = Random.MersenneTwister(1)
		X = rand(rng, 3, 2)
		Φ = get_features(pm, X)
		Φ_true = [1.0 0.23603334566204692 0.00790928339056074 0.05571174026441932 0.0018668546204633095 6.25567637522e-5 0.013149828447265864 0.00044063994193260575 1.4765482242222028e-5 4.947791725125075e-7; 1.0 0.34651701419196046 0.4886128300795012 0.12007404112451132 0.16931265897503248 0.2387424977182995 0.04160769821242834 0.058669717052929886 0.08272833747007607 0.11665264747038717; 1.0 0.3127069683360675 0.21096820215853596 0.09778564804593431 0.06597122691230639 0.04450758232200489 0.03057825354722182 0.020629662365158116 0.013917831135882103 0.00938968462489641]
		check_1 = check_elements(Φ_true, Φ) #all(Φ .≈ Φ_true)
		pm = Polynomial3Model(2, 1; ignore_first=true)
		X_bias = ones(size(X, 1), size(X, 2) + 1)
		X_bias[:, 2:end] .= X
		Φ = get_features(pm, X_bias)
		check_2 = check_elements(Φ_true, Φ)#all(Φ .≈ Φ_true)
		check_3 = (size(Φ)==size(Φ_true))
		check_1 && check_2 && check_3
	end
	
	HTML("<h2 id=poly> (a) Polynomial Features $(_check_complete(__check_Poly2_logit_reg))</h4>")
end

# ╔═╡ 0608c93d-2f82-470c-8d9f-ca79af6b2612
predict(lr::Polynomial3Model, X) = predict(lr.model, get_features(lr, X))

# ╔═╡ d9935cc8-ec24-47e9-b39a-92c21377a161
struct MiniBatchGD
	b::Int
end

# ╔═╡ e7712bd3-ea7e-4f4a-9efc-041b4b2be987
begin
	"""
		RandomModel
	
	Predicts `w*x` where `w` is sampled from a normal distribution.
	"""
	struct RandomModel <: AbstractModel # random weights
		W::Matrix{Float64}
		γ::Float64 # Threshold on binary classification confidence
	end
	RandomModel(in, out) = RandomModel(randn(in, out), 0.5)
	# predict(logit::RandomModel, X::AbstractMatrix) = sigmoid(X*logit.W) .>= Array(logit.γ, length(X*logit.W), 1) ? 1.0 : 0.0
	Base.copy(logit::RandomModel) = RandomModel(randn(size(logit.W)...), logit.γ)
	train!(::MiniBatchGD, model::RandomModel, lossfunc, opt, X, Y, num_epochs) = 
		nothing
end;

# ╔═╡ d77fe746-6fca-4a9e-97ac-0066db0ed2ca
function predict(logit::RandomModel, X::AbstractMatrix)
	Ŷ = sigmoid(X*logit.W) 
	pred = zeros(size(Ŷ))
	for i in 1:length(Ŷ)
		if Ŷ[i] >= logit.γ
			pred[i] = 1.0
		else
			pred[i] = 0.0
		end
	end
	pred
end;

# ╔═╡ 5714c84f-1653-4c4a-a2e4-003d8560484a
md"""
In this notebook, we will be focusing on minibatch gradient descent. Below you need to (re)implement the function `epoch!`. You can just use your code from Assignment 3 on `MBGD`. **We are not grading this section again, since it is the same code as before. But, if it is incorrect, it will give wrong results in later sections, resulting in deductions there. Please ensure this code is correct.** This function should go through the data set in mini-batches of size `mbgd.b`. Remember to randomize how you go through the data **and** that you are using the correct targets for the data passed to the learning update. In this implementation, you will use 

```julia
update!(model, lossfunc, opt, X_batch, Y_batch)
```

to update your model. You randomize and divide the dataset into batches and call the update function for each batch. These update functions are defined in the section on [optimizers](#opt).

"""

# ╔═╡ 9d96ede3-533e-42f7-ada1-6e71980bc6c2
function epoch!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y)
	epoch!(mbgd, get_linear_model(model), lossfunc, opt, get_features(lp.model, X), Y)
end;

# ╔═╡ 6ff92fca-6d66-4f27-8e09-11a3887e66ba
function train!(mbgd::MiniBatchGD, model::AbstractModel, lossfunc, opt, X, Y, num_epochs)
	train!(mbgd, get_linear_model(model), lossfunc, opt, get_features(model, X), Y, num_epochs)
end;

# ╔═╡ 7e777dba-b389-4549-a93a-9b0394646c57
abstract type LossFunction end

# ╔═╡ f380a361-2960-471b-b29a-3bd1fe06252b
md"""
#### (c) Cross-entropy
"""

# ╔═╡ 6d2d24da-9f3f-43df-9243-fc17f85e0b01
md"""
The cross-entropy loss function for the whole dataset is

```math
c(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n c_i(\mathbf{w})
```
for cross-entropy loss $c_i$ on datapoint $i$
```math
c_i(\mathbf{w}) = −y_i  \ln  σ(\mathbf{x}_i^T \mathbf{w}) - (1 − y_i) \ln(1 − \sigma(\mathbf{x}_i^T \mathbf{w}))
```

You should be using the sigmoid function $\sigma$ that you implemented above. When computing the loss for a minibatch $\{i_1, i_2, \ldots, i_b\}$, we sum losses $c_i$ for those points

```math
\frac{1}{b}\sum_{j=1}^b c_{i_j}(\mathbf{w})
```
You will compute the loss for a given batch of data, inputted into `loss` as `X` and `Y`. This data could be the entire dataset, or a minibatch. 

"""

# ╔═╡ 4f43373d-42ee-4269-9862-f53695351ea3
struct CrossEntropy <: LossFunction end

# ╔═╡ ada800ba-25e2-4544-a297-c42d8b36a9ff
function loss(lm::AbstractModel, ce::CrossEntropy, X , Y)
	θ = predict(lm, X) # θ = Xw
	loss = 0.0
	#### BEGIN SOLUTION
	for i in 1:length(Y)
		sigma = sigmoid(transpose(X[i,:]) * lm.w)
		loss += (-Y[i] * log(sigma[1])) - ((1 - Y[i]) * log(1 - sigma[1]))	
	end
	return loss/ length(Y)
	#### END SOLUTION
end;

# ╔═╡ 7bea0c90-077f-4eca-b880-02d1289244f3
md"""
#### (d) Gradient of Cross-Entropy
"""

# ╔═╡ 4ea14063-99ca-4caf-a862-fbf9590c68a2
md"""
You will implement the gradient of the cross-entropy loss function in the `gradient` function, returning a matrix of the same size of `lm.w` using the following formula:
```math
\nabla c_i(\mathbf{w}) = (σ(\mathbf{x}_i^T \mathbf{w}) − y_i)\mathbf{x}_i
```

"""

# ╔═╡ 299116ea-66f3-4e52-ab0f-594249b9dd23
function gradient(lm::AbstractModel, ce::CrossEntropy, X::Matrix, Y::Vector)
	∇w = zero(lm.w) # gradients should be the size of the weights
	
	#### BEGIN SOLUTION
    for i in 1: length(Y)
		sigma = sigmoid(transpose(X[i,:]) * lm.w)
        ∇w += (sigma[1] - Y[i])*X[i, :]
    end
    return ∇w/length(Y)
	#### END SOLUTION

	@assert size(∇w) == size(lm.w)
	∇w
end;

# ╔═╡ 36c1f5c8-ac43-41ea-9100-8f85c1ee3708
abstract type Optimizer end

# ╔═╡ 159cecd9-de77-4586-9479-383661bf3397
begin
	struct _LR <: Optimizer end
	struct _LF <: LossFunction end
	function gradient(lm::LinearModel, lf::_LF, X::Matrix, Y::Vector)
		sum(X, dims=1)
	end
	function update!(lm::LinearModel, 
		 			 lf::_LF, 
		 			 opt::_LR, 
		 			 x::Matrix,
		 			 y::Vector)
		
		ϕ = get_features(lm, x)
		
		Δw = gradient(lm, lf, ϕ, y)[1, :]
		lm.w .-= Δw
	end
end;

# ╔═╡ a3387a7e-436c-4724-aa29-92e78ea3a89f
begin
	_X = [4 3 4 1; 1 0 5 1; 1 5 6 1; 4 4 7 1; 2 4 8 1]
	__check_cegrad = all(gradient(LinearModel(4, 1), CrossEntropy(), _X, [1.0,0.0,0.0,1.0,1.0]) .≈ [-0.8; -0.6000000000000001; -0.8; -0.1])
	__check_celoss = loss(LinearModel(4, 1), CrossEntropy(), _X, [1.0,0.0,0.0,1.0,1.0]) .≈ 0.6931471805599454
	
	__check_CrossEntropy = __check_celoss && __check_cegrad
	
md"""
For this notebook we only use the cross-entropy, but we still use the abstract type LossFunction as a standard abstract type for all losses. Below you will need to implement the `loss` $(_check_complete(__check_celoss)) function and the `gradient` $(_check_complete(__check_cegrad)) function for Cross_Entropy.
"""
end

# ╔═╡ a17e5acd-d78d-4fab-9ab2-f01bd888339d
HTML("<h3 id=lossfunc> Loss Functions $(_check_complete(__check_CrossEntropy)) </h3>")

# ╔═╡ 77cda37c-c560-42d8-85b1-7f026d09edfe
md"""
RMSprop is another adaptive learning rate that uses a different learning rate for every parameter $w_j$. Instead of taking cumulative sum of squared gradients as in AdaGrad, we take the exponential moving average of these gradients. The motivation for doing so is to allow the gradient to stay larger for longer, if needed. The stepsize for Adagrad can decay quite quickly, due to the accumulation of squared gradients. 

To implement RMSprop, we use the following equations:

```math
\begin{align}
\bar{\mathbf{g}}_{t+1} &= \beta \bar{\mathbf{g}}_{t} + (1-\beta) \mathbf{g}_t^2 \\
\mathbf{w}_{t+1} &= \mathbf{w}_{t} - \frac{\eta}{\sqrt{\bar{\mathbf{g}}_{t+1} + \epsilon}}  \mathbf{g}_t
\end{align}
```
where $\mathbf{g}_t$ is the gradient at time step $t$ and the addition, squaring, multiplication and division are all elementwise. The coefficient $\beta$ represents the degree of weighting decrease, a constant smoothing factor between $0$ and $1$. A higher $\beta$ discounts older observations faster.

Implement ```RMSprop``` below.
"""

# ╔═╡ 1fe7084a-9cf9-48a4-9e60-b712365eded9
begin
	mutable struct RMSprop <: Optimizer
		η::Float64 # step size
		β::Float64 # The significance coefficient on the most recent data points
		gbar::Matrix{Float64} # exponential decaying average
		ϵ::Float64 # 
	end
	
	RMSprop(η) = RMSprop(η, 0.9, zeros(1, 1), 1e-5)
	RMSprop(η, lm::LinearModel) = RMSprop(η, 0.9, zero(lm.w), 1e-5)
	RMSprop(η, model::AbstractModel) = RMSprop(η, get_linear_model(model))
	Base.copy(rmsprop::RMSprop) = RMSprop(rmsprop.η, rmsprop.β, zero(rmsprop.gbar), rmsprop.ϵ)
end

# ╔═╡ c2710a60-ebe1-4d01-b6d1-0d6fe45723f9
function update!(lm::LinearModel, 
				 lf::LossFunction,
				 opt::RMSprop,
				 x::Matrix,
				 y::Vector)

	g = gradient(lm, lf, x, y)
	if size(g) !== size(opt.gbar) # need to make sure this is of the right shape.
		opt.gbar = zero(g)
	end
	
	# update opt.gbar and lm.w	
	#### BEGIN SOLUTION
	for i in 1: length(lm.w)
		opt.gbar[i] = opt.β * opt.gbar[i] + (1- opt.β) * g[i]^2
		lm.w[i] = lm.w[i] - (opt.η * 1/sqrt(opt.gbar[i] + opt.ϵ)) * g[i]
	end
	
	#### END SOLUTION	
end;

# ╔═╡ 69cf84e2-0aba-4595-8cb0-c082dbccdbe2
function epoch!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y)
	
	#### BEGIN SOLUTION
	start = 1
	batch_size = mbgd.b
	randindex = randperm(length(Y))
	for i in 1:floor(Int, size(X,1)/ mbgd.b)
		X_b = X[randindex[start : batch_size], :]
		Y_b = Y[randindex[start : batch_size]]
		update!(model, lossfunc, opt, X_b, Y_b)
	end
	#### END SOLUTION
end;

# ╔═╡ acf1b36c-0412-452c-ab4d-a388f84fd1fb
begin
	__check_MBGD = let

		lm = LinearModel(3, 1)
		opt = _LR()
		lf = _LF()
		X = ones(10, 3)
		Y = collect(0.0:0.1:0.9)
		mbgd = MiniBatchGD(5)
		epoch!(mbgd, lm, lf, opt, X, Y)
		all(lm.w .== -10.0)
	end
	str = "<h2 id=graddescent> (b) Mini-batch Gradient Descent $(_check_complete(__check_MBGD)) </h2>"
	HTML(str)
end

# ╔═╡ 2782903e-1d2e-47de-9109-acff4595de42
function train!(mbgd::MiniBatchGD, model::LinearModel, lossfunc, opt, X, Y, num_epochs)
	ℒ = zeros(num_epochs + 1)
	ℒ[1] = loss(model, lossfunc, X, Y)
	for i in 1:num_epochs
		epoch!(mbgd, model, lossfunc, opt, X, Y)
		ℒ[i+1] = loss(model, lossfunc, X, Y)
	end
	ℒ
end;

# ╔═╡ 8dfd4734-5648-42f2-b93f-be304b4b1f27
begin
	 __check_RMSprop_v, __check_RMSprop_W = let
		lm = LinearModel(2, 1)
		opt = RMSprop(0.1, lm)
		X = [0.1 0.5; 
			 0.5 0.0; 
			 1.0 0.2]
		Y = [1, 0, 1]
		update!(lm, CrossEntropy(), opt, X, Y)
		true_G = [0.0009999999999999996; 0.0013611111111111105]
		true_W = [0.31465838776377636; 0.31507247500483543]
		all(opt.gbar .≈ true_G), all(lm.w .≈ true_W)
	end
	
	__check_RMSprop = __check_RMSprop_v && __check_RMSprop_W
	
md"""
#### (f) RMSprop $(_check_complete(__check_RMSprop))

	
"""
end

# ╔═╡ af8acfdf-32bd-43c1-82d0-99008ee4cb3e
HTML("<h3 id=opt> Optimizers $(_check_complete(__check_RMSprop)) </h3>")

# ╔═╡ 3738f45d-38e5-415f-a4e6-f8922df84d09
md"""
Below you will need to implement an optimizer:

- RMSprop $(_check_complete(__check_RMSprop))
"""

# ╔═╡ fa610de0-f8c7-4c48-88d8-f5398ea75ae2
md"""
# Evaluating models

In the following section, we provide a few helper functions and structs to make evaluating methods straightforward. The abstract type `LearningProblem` with child `GDLearningProblem` is used to construct a learning problem. Once again, we introduce an abstract type even though we only have one child, but provide a design that does not constrain the types of learners that are possible. This struct contain all the information needed to `train!` a model for gradient descent. We also provide the `run` and `run!` functions. These will apply the feature transform (if there is one) and train the model. `run` does this with a copy of the learning problem, while `run!` does this inplace. 

"""

# ╔═╡ d695b118-6d0d-401d-990f-85ba467cc53e
abstract type LearningProblem end

# ╔═╡ 6edc243e-59ac-4c6f-b507-80d3ec13bc21
"""
	GDLearningProblem

This is a struct for keeping a the necessary gradient descent learning setting components together.
"""
struct GDLearningProblem{M<:AbstractModel, O<:Optimizer, LF<:LossFunction} <: LearningProblem
	gd::MiniBatchGD
	model::M
	opt::O
	loss::LF
end;

# ╔═╡ 3bdde6cf-3b68-46d3-bf76-d68c20b661e9
Base.copy(lp::GDLearningProblem) = 
	GDLearningProblem(lp.gd, copy(lp.model), copy(lp.opt), lp.loss)

# ╔═╡ 7905f581-1593-4e06-8aaf-faec05c3b306
function run!(lp::GDLearningProblem, X, Y, num_epochs)
	update_transform!(lp.model, X, Y)
	train!(lp.gd, lp.model, lp.loss, lp.opt, X, Y, num_epochs)
end;

# ╔═╡ 69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
function run(lp::LearningProblem, args...)
	cp_lp = copy(lp)
	ℒ = run!(cp_lp, args...)
	return cp_lp, ℒ
end;

# ╔═╡ 1cdb6679-c18f-46f7-8f23-9ed6e138a7a9
md"""
### Accuracy and Misclassification Error
The Accuracy of a model is the percent correct on the given (testing) data, namely the percent of points where we predicted the correct class. The misclassification error is 100 minus this number, shows the percent incorrect. The misclassification error on the test set is an estimate of the expected 0-1 cost of the classifier, given as a percent rather than a percentage between 0 and 1. 
"""

# ╔═╡ 89cc730e-ab66-4f87-827c-87539ac1f54a
function get_accuracy(Y, Ŷ)
    correct = 0
    # count number of correct predictions
    correct = sum(Y .== Ŷ)
    # return percent correct
    return (correct / Float64(length(Y))) * 100.0
end;

# ╔═╡ 045b8be8-58c6-497b-baac-8af41de76b1e
function get_misclassification_error(Y, Ŷ)
    return (100 - get_accuracy(Y, Ŷ))
end;

# ╔═╡ eef918a9-b8af-4d41-85b1-bebf1c7889cc
HTML("<h4 id=cv> Run Experiment </h2>")

# ╔═╡ fd75ff49-b5de-48dc-ae89-06bf855d81b2
md"""

Below are the helper functions for running an experiment.

"""

# ╔═╡ d339a276-296a-4378-82ae-fe498e9b5181
"""
	run_experiment(lp, X, Y, num_epochs, runs; train_size)

Using `train!` do `runs` experiments with the same train and test split (which is made by `random_dataset_split`). This will create a copy of the learning problem and use this new copy to train. It will return the estimate of the error.
"""
function run_experiment(lp::LearningProblem, 
						data,	 
						num_epochs,
						runs)
	err = zeros(runs)

	X, Y = data.X, data.Y
	train_size=20000
	test_size = 1000
	
	for i in 1:runs		
		
		rp = randperm(length(Y))
		train_idx = rp[1:train_size]
		test_idx = rp[train_size+1:train_size+test_size]
		train_data = (X[train_idx, :], Y[train_idx]) 
		test_data = (X[test_idx, :], Y[test_idx])
		
		# train
		cp_lp, train_loss = run(lp, train_data[1], train_data[2], num_epochs)
		
		# test
		Ŷ = predict(cp_lp.model, test_data[1])
		err[i] = get_misclassification_error(test_data[2], Ŷ)
	end

	err
end;

# ╔═╡ 58e626f1-32fb-465a-839e-1f413411c6f3
md"
# Experiments

In this section, we will run an experiment on the algorithms we implemented above. We provide the data in the `Data` section, and then follow the experiment and its description. You will need to analyze and understand the experiment for the written portion of this assignment.
"

# ╔═╡ 5ec88a5a-71e2-40c1-9913-98ced174341a
md"""
## The Dataset

We use a the physics dataset, from the UCI repository. We normalize the columns using min-max scaling. A description of the dataset is given below, given by the group that released the dataset. 
"""

# ╔═╡ d2c516c0-f5e5-4476-b7d6-89862f6f2472
function unit_normalize_columns!(df::DataFrame)
	for name in names(df)
		mn, mx = minimum(df[!, name]), maximum(df[!, name])
		df[!, name] .= (df[!, name] .- mn) ./ (mx - mn)
	end
	df
end;

# ╔═╡ 90f34d85-3fdc-4e2a-ada4-085154103c6b
physiscs_data = let
	data = CSV.read("data/susysubset.csv", DataFrame, delim=',', ignorerepeated=true)[:, 1:end]
	data[!, 1:end-1] = unit_normalize_columns!(data[:, 1:end-1])
	data
end;

# ╔═╡ eac4fb9d-126b-4ba8-9078-105638416de2
md"""
Here is a description of the physics dataset in case you are interested.
```
The data has been produced using Monte Carlo simulations and contains events with two leptons (electrons or muons). In high energy physics experiments, such as the ATLAS and CMS detectors at the CERN LHC, one major hope is the discovery of new particles. To accomplish this task, physicists attempt to sift through data events and classify them as either a signal of some new physics process or particle, or instead a background event from understood Standard Model processes. Unfortunately we will never know for sure what underlying physical process happened (the only information to which we have access are the final state particles). However, we can attempt to define parts of phase space that will have a high percentage of signal events. Typically this is done by using a series of simple requirements on the kinematic quantities of the final state particles, for example having one or more leptons with large amounts of momentum that is transverse to the beam line ( pT ). Here instead we will use logistic regression in order to attempt to find out the relative probability that an event is from a signal or a background event and rather than using the kinematic quantities of final state particles directly we will use the output of our logistic regression to define a part of phase space that is enriched in signal events. The dataset we are using has the value of 18 kinematic variables ("features") of the event. The first 8 features are direct measurements of final state particles, in this case the  pT , pseudo-rapidity ( η ), and azimuthal angle ( ϕ ) of two leptons in the event and the amount of missing transverse momentum (MET) together with its azimuthal angle. The last ten features are functions of the first 8 features; these are high-level features derived by physicists to help discriminate between the two classes. You can think of them as physicists attempt to use non-linear functions to classify signal and background events and they have been developed with a lot of deep thinking on the part of physicist. There is however, an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks and the dropout algorithm are presented in the original paper to compare the ability of deep-learning to bypass the need of using such high level features. We will also explore this topic in later notebooks. The dataset consists of 5 million events, the first 4,500,000 of which we will use for training the model and the last 500,000 examples will be used as a test set.
```
"""

# ╔═╡ 14b329fb-8053-4148-8d24-4458e592e7e3
md"""
## Plotting our results

The `plot_results` function produces two plots. The left plot is a box plot over the errors, the right plot is a bar graph displaying average errors with standard error bars. This function will be used for all the experiments, and you should use this to finish your written experiments.

"""


# ╔═╡ eebf5285-2336-4c07-a4fd-b1fd841dee52
function plot_results(algs, errs; vert=false)
	stderr(x) = sqrt(var(x)/length(x))
	
	plt1 = boxplot(reshape(algs, 1, :),
				   errs,
				   legend=false, ylabel="Misclassification error",
				   pallette=:seaborn_colorblind)
	
	plt2 = bar(reshape(algs, 1, :),
			   reshape(mean.(errs), 1, :),
			   yerr=reshape(stderr.(errs), 1, :),
			   legend=false,
			   pallette=:seaborn_colorblind,
			   ylabel=vert ? "Misclassification error" : "")
	
	if vert
		plot(plt1, plt2, layout=(2, 1), size=(600, 600))
	else
		plot(plt1, plt2)
	end
end;

# ╔═╡ 9ed07108-2ed0-430f-ab97-6f51297c5361
md"""
## (g) Evaluating the Classifiers

We will compare different classifiers on the [Physics dataset](). 

To run this experiment click $(@bind __run_class PlutoUI.CheckBox())

**You can get the misclassification error to report from the plot or from the terminal where your ran this notebook**.

Note that it might take a bit of time to run this experiment (even a few minutes), so be a bit patient waiting for the graph to appear right below this text.
"""

# ╔═╡ d686c8ca-cb29-4f7c-8872-a907173b156c
begin
	if __run_class
		algs = ["Random", "Logit", "PolyLogit"]
		classification_problems = [
			GDLearningProblem(
				MiniBatchGD(200),
				RandomModel(8, 1),
				RMSprop(0.01),
				CrossEntropy()),
			GDLearningProblem(
				MiniBatchGD(200),
				LogisticRegressor(8, 1),
				RMSprop(0.01),
				CrossEntropy()),
			GDLearningProblem(
				MiniBatchGD(200),
				Polynomial3Model(8, 1),
				RMSprop(0.01),
				CrossEntropy())
			];
		
		misclass_errs = let
			Random.seed!(2)
			data = (X=Matrix(physiscs_data[:, 1:end-1]), Y=physiscs_data[:, end])
			@show size(data.X)
			errs = Vector{Float64}[]
			
			for (idx, prblms) in enumerate(classification_problems)

				err = run_experiment(prblms, data, 100, 10)
				push!(errs, err)
			end
			errs
		end

		mean_error_Random = mean(misclass_errs[1])
		mean_error_Logit = mean(misclass_errs[2])
		mean_error_PolyLogit = mean(misclass_errs[3])
		
		println("Misclassification error on the test set for Random model is $mean_error_Random.")
		
		println("Misclassification error on the test set for Logistic Regression model is $mean_error_Logit.")
		
		println("Misclassification error on the test set for Polynomial Logistic Regression model is $mean_error_PolyLogit.")
		
		plot_results(algs, misclass_errs)
	end

end

# ╔═╡ 14fa89f8-d034-4286-bdb2-2c11190e17d0
md"""
## Q2: Hypothesis Testing
"""

# ╔═╡ 26466a2e-5554-407f-8729-e2b841f10a7e
md"""

In this question, you will use the paired t-test to compare the performance of two models. You will compare the two models from above (logistic regression and polynomial logistic regression) both using RMSprop for optimization. You hypothesize that polynomial logistic regression is better than logistic regression and you want to run a one-tailed t-test to see if this is true. 

The paired t-test is designed for continuous errors, rather than 0-1 errors. For 0-1 errors, and one test set, we often use McNemar's test. However, there is another alternative. We can run the algorithm on multiple random training and testing splits (say 10) to get 10 estimates of misclassification error (average 0-1 error on a test set, as a percent). The standard deviation and average of these 10 errors provides a useful insight into the quality of the model we would get for this problem. We therefore run each algorithm 10 times, on random training and test splits. We control the splits so that both algorithms train and test on the same split, to have paired samples for the 10 samples of error. We then conduct a paired t-test on these 10 numbers. 

The misclassification error estimate above, in fact, is the average of these 10 numbers. It looks like polynomial logistic regression is better than logistic regression. Now we will examine whether we can say this with confidence (that the difference is statistically significant). 

"""

# ╔═╡ 1189ddd5-9295-4e1c-a50e-11efed56d35b
md"""
### (a) Defining the Null Hypothesis
"""

# ╔═╡ 9e33cb14-44b6-4335-a899-7d51a9829346
md"""
Define the null hypothesis and the alternative hypothesis. Assume $\mu_1$ to be the true expected misclassification error for Logistic Regression and $\mu_2$ to be the true expected misclassification error for Polynomial Logistic Regression. This is not code that is run, rather it is simply your written answer to your question. For marking purposes, it is easier for us if you to write this description here. Note that to write these answers, you have to put them in comments because they are not valid Julia code. 
"""

# ╔═╡ 11098780-5235-40a1-9477-091ce68420a9
# discussion should go here
#### BEGIN SOLUTION

# Null Hypothesis: Mu_1 > Mu_2, that is Polynomial logistic regression is better than logistic regression as the mean of pol. logistic regression Mu_2 is lower than Mu_1
# Alternative Hypothesis: Mu_1 > Mu_2 does not hold

#### END SOLUTION

# ╔═╡ 3468f550-da21-4e80-a030-0ab74439c1ee
md"""

Before running the paired t-test, you should check that the assumptions are not violated. One way to satisfy the assumption for the paired t-test is to check that the errors are (approximately) normally distributed with (approximately) equal variances. The Student’s t-distribution is approximately like a normal distribution, with a degrees-of-freedom parameter $m-1$ that makes the distribution look more
like a normal distribution as m becomes larger.

To do this, you need to implement the ```checkforPrerequisites``` method below. For each model, you can plot a
histogram of its errors on the test set. You can do so by using the two vectors of errors and the function ```plot_histogram``` function to visualize the error distributions simultaneously. Discuss why it is ok or not ok to use the paired t-test to get statistically sound conclusions about these two models. From Q1, you will use the ```logisticRegression_error``` as the ```baseline_error``` and ```PolynomialLogisticRegression_error``` as the ```learner_error```.

Once again, this is not code that is run. Rather, we will read your comments. Note that this answer need not be long; a few sentences suffices. 
"""

# ╔═╡ fa0bac24-e5c5-425b-b719-a0b98dead6b2
# discussion should go here
#### BEGIN SOLUTION

#### END SOLUTION

# ╔═╡ e2038762-570f-461b-8f39-0549446d1e5b
function plot_histogram(baseline_error::AbstractVector{Float64},
	learner_error::AbstractVector{Float64})
	histogram([learner_error baseline_error], label = ["learner_error" "baseline_error"])
end;

# ╔═╡ 4277a0fb-3746-424c-80af-02f6862c3258
function checkforPrerequisites(baseline_error::AbstractVector{Float64},
		learner_error::AbstractVector{Float64},
		learner_name::AbstractString,
		baseline_name::AbstractString)
	# Compute mean and std of the error distributions and plot their histograms
	
	mu_1, mu_2, std_1, std_2 = 0.0, 0.0, 0.0, 0.0
	#### BEGIN SOLUTION
	mu_1 = mean( baseline_error)
	std_1 = std(baseline_error)
	
	mu_2 = mean( learner_error)
	std_2 = std(learner_error)
	
	# plot_histogram(baseline_error, learner_error)

	#### END SOLUTION
	
	println("BaseLine Error: mean = $mu_1 and Standard deviation = $std_1")
	println("Learner Error: mean = $mu_2 and Standard deviation = $std_2")
	
	mu_1, mu_2, std_1, std_2
	
end;

# ╔═╡ 4f8d8553-dfaa-4d49-a5b9-c0993fb03e85
begin
	__check_q2b = let
		e1 = [30.5, 44.3, 46.8, 50.9]
		e2 = [50.5, 49.3, 51.8, 55.5]
		mu_1, mu_2, std_1, std_2 = checkforPrerequisites(e1, e2, "e1", "e2")
		mu_1 == 43.125 && std_1 == 8.845479071254422 && mu_2 == 51.775 && std_2 == 2.6849891371599015 
	end
	HTML("<h2 id=checking_assumptions> (b) Checking for Assumptions $(_check_complete(__check_q2b))")
end

# ╔═╡ 00cf8e83-4953-4487-814c-7c6cd11b5e11



# ╔═╡ 91636720-2e59-47ab-87ff-a50e9d817d18
md"""

Regardless of the outcome of (b), let’s run the one-tailed t-test. (Note, I am not
advocating that you check for violated assumptions and then ignore the outcome of that step. The goal of this question is simply to give you experience actually running a statistical significance test. Presumably, in practice, you would pick an appropriate one after verifying assumptions). 

To run this test, you need to compute the p-value. To do this implement the ```getPValue``` method, which returns the p-value for the one-tailed paired t-test. It looks at the positive part of the tail: we consider the difference between the baseline and learner, where if it is a larger positive number that the error for the learner is much lower than the baseline. This test therefore checks if the learner (polynomial logistic regression) is statistically significantly better than the baseline (logistic regression). 

Report the p-value. Would you be able to reject the null hypothesis with a significance threshold of 0.05? How about of 0.01? Include both the p-value as well as this discussion in the comments right below this text.
"""

# ╔═╡ 015bdde5-77c9-49d7-a8ba-c9c11f0033a9
# discussion should go here
#### BEGIN SOLUTION


#### END SOLUTION

# ╔═╡ 2cd21c3b-c826-40d6-9bdf-5e808c23a7d3
# helper function to get the positive tail p-value using t-distribution
function pValueTDistPositiveTail(t::Float64, dof::Int64)
	1 - cdf(TDist(dof), t)
end;

# ╔═╡ c71c562b-bb4f-4688-a451-daf2040ede62
function tDistPValue(baseline_error::AbstractVector{Float64}, 		           learner_error::AbstractVector{Float64})
	# Computes the p-value using one-tailed t-test
	@assert size(learner_error) == size(baseline_error)
	m = size(learner_error, 1) # the number of test samples
	dof = m - 1
	t = 0.0
	#### BEGIN SOLUTION
	s = 0.0
	d = zeros(size(learner_error))
	# println(d)
	d = baseline_error - learner_error
	d_bar = sum(d) / m
	s_d = sqrt(sum((d .- d_bar).^2)/dof)
	t = d_bar/(s_d/sqrt(m))

	# println(1 - cdf(TDist(dof), t))
	#### END SOLUTION
	pValueTDistPositiveTail(t, dof)
end;

# ╔═╡ f8ff4eca-1d38-45dc-9d85-5e0de774aa4a
begin
	__check_q2c = let
		e1 = [30.5, 44.3, 46.8, 50.9]
		e2 = [50.5, 49.3, 51.8, 55.5]
		pval = tDistPValue(e1, e2)
		pval == 0.946808781297146
	end
	HTML("<h2 id=t_test> (c) Running the t-test $(_check_complete(__check_q2c))")
end

# ╔═╡ a9d3c6c3-1cb7-4417-ba6a-54998841c87d
let
	q1_a_check = _check_complete(__check_logit_reg)
	q1_b_check = _check_complete(__check_Poly2_logit_reg)
	q1_c_check = _check_complete(__check_MBGD)
	q1_d_check = _check_complete(__check_celoss)
	q1_e_check = _check_complete(__check_cegrad)
	q1_f_check = _check_complete(__check_RMSprop)

md"""
# Preamble
	
In this assignment, we will implement:
	
- Q1(a) [Logistic Regression: sigmoid function](#logit) $(q1_a_check)
- Q1(b) [Polynomial Logistic Regression](#graddescent) $(q1_b_check)
- Q1(c) [Mini-batch Gradient Descent](#gd) use the old code in Assignment 3 $(q1_c_check) 
- Q1(d) [Loss Function](#lossfunc) cross entropy $(q1_d_check)
- Q1(e) [Gradient of Loss Function](#gradlossfunc) gradient of cross entropy $(q1_e_check)
- Q1(f) [Optimizer](#opt): adapive stepsize RMSprop $(q1_f_check)
- Q2(a) Hypothesis-testing: Define Null hypothesis and alternative hypothesis
- Q2(b) [Checking for assumptions](#checking_assumptions): before running the t-test $(__check_q2b)
- Q2(c) [Running the t-test](#t_test): get the pvalue and run the t-test $(__check_q2c)
"""
end

# ╔═╡ efe707a1-41c7-439d-9e57-dfea374f355d
md"""
Next, you will run the t_test given the functions implemented, then you can tell whether we reject the null hypothesis or not.
"""

# ╔═╡ 62a4c01b-434b-4a2b-a46b-78bee5136dad
function t_test(baseline_error::AbstractVector{Float64},
		learner_error::AbstractVector{Float64},
		learner_name::AbstractString, 
		baseline_name::AbstractString, 
		pvalueThreshold::Float64)

	checkforPrerequisites(baseline_error, learner_error, baseline_name, learner_name)
    pval = tDistPValue(baseline_error, learner_error)
	
    if pval < pvalueThreshold
        result = "rejected"
    else
        result = "not rejected"
	end
    println("With pvalue = $pval, the null hypothesis is $result under a pvalueThreshold of $pvalueThreshold")
end;

# ╔═╡ 07314336-5096-4fd3-a94a-70e41a7d3a6a
begin
	if __run_class
		baseline_error = misclass_errs[2]
		learner_error =  misclass_errs[3]
		
		baseline_name = "LogisticRegression"
		learner_name = "PolynomialLogisticRegression"
	
		pvalueThreshold = 0.05
	
		t_test(baseline_error, learner_error, learner_name, baseline_name, pvalueThreshold)
	end
end

# ╔═╡ f3bd26ee-ea95-4590-aa7c-da5f293b2d77
begin
	if __run_class
		e1 = misclass_errs[2]
		e2 = misclass_errs[3]
	
		checkforPrerequisites(baseline_error, learner_error, "LogisticRegression", "PolynomialLogisticRegression")
		plot_histogram(e1, e2)
		
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.9.11"
DataFrames = "~1.3.0"
Distributions = "~0.25.34"
MultivariateStats = "~0.8.0"
PlutoUI = "~0.7.21"
StatsBase = "~0.33.13"
StatsPlots = "~0.14.29"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "49f14b6c56a2da47608fe30aed711b5882264d7a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.11"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4c26b4e9e91ca528ea212927326ece5918a04b47"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.2"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "2e993336a3f68216be91eb8ee4625ebbaba19147"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "7f3bec11f4bcd01bc1f507ebce5eadf1b0a78f47"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.34"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "30f2b340c2fff8410d89bfcdc9c0a6dd661ac5f7"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.62.1"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fd75fa3a2080109a2c0ec9864a6e14c60cca3866"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.62.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "ca99cac337f8e0561c6a6edeeae5bf6966a78d21"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a8f4f279b6fa3c3c4f1adadd78a621b13a506bce"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.9"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun"]
git-tree-sha1 = "8789439a899b77f4fbb4d7298500a6a5781205bc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.25.0"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "b68904528fd538f1cb6a3fbc44d2abdc498f9e8e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.21"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "f45b34656397a1f6e729901dc9ef679610bd12b5"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f0bccf98e16759818ffc5d97ac3ebf87eb950150"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[StatsFuns]]
deps = ["ChainRulesCore", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "bedb3e17cc1d94ce0e6e66d3afa47157978ba404"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.14"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "d6956cefe3766a8eb5caae9226118bb0ac61c8ac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.29"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "66d72dc6fcc86352f01676e8f0f698562e60510f"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.23.0+0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─75441ce6-2137-4fcf-bba2-6ed67b9acb59
# ╠═dc004086-db6e-4813-841e-d427520402f7
# ╠═c1b9d7a3-5605-4478-b1a9-a9553ad38ed7
# ╠═c9797979-7d16-4ba4-8f82-c7ec33f6e517
# ╟─693a3933-c1c2-4249-8c03-f5151267222f
# ╟─bc0c5abe-1c2a-4e5c-bb0c-8d76b5e81133
# ╠═def97306-1703-42bc-bc09-da623c545e87
# ╟─14c30b50-157f-40bf-b0a6-0232aa2705c6
# ╠═4a196e82-fe94-49fe-ab87-d57e2d7fbd34
# ╟─a9d3c6c3-1cb7-4417-ba6a-54998841c87d
# ╠═7247e88f-e9c3-4561-ae67-71eb23ad066d
# ╟─a7aecd21-13f2-4cf2-8239-3a3d708602c9
# ╟─e3c3e903-e2c2-4de0-a3b2-56a27709e8c3
# ╟─a35944ae-7bbb-4f3c-bbe0-d2bbb57cd70b
# ╠═e7712bd3-ea7e-4f4a-9efc-041b4b2be987
# ╠═d77fe746-6fca-4a9e-97ac-0066db0ed2ca
# ╟─7b513344-1cad-4eef-9faf-e77ba176323e
# ╟─4f4029a2-c590-4bd3-a0db-d2380d4b4620
# ╟─dcfecc35-f25c-4856-8429-5c31d94d0a42
# ╠═d45dedf8-e46a-4ffb-ab45-f9d211d3a8ca
# ╠═7cd46d84-a74c-44dc-8339-68010924bc39
# ╠═8745fec1-47c8-428b-9ea4-1e6828618830
# ╟─fd4df2c6-fbfe-4089-a487-e3965a463ef3
# ╠═2d43a3ba-2a2c-4114-882f-5834d42e302a
# ╟─9c0491db-ad27-46fc-969c-c42a14cdadeb
# ╠═0005ee8a-fcef-4c4e-8aa5-6140b06ed4ef
# ╠═51599971-4638-4787-9c13-75afa0d34285
# ╠═50cb6e7f-3341-47b8-b720-d069f03f1be2
# ╠═8847c374-c9f4-470d-80be-2946f2af9661
# ╟─c59cf592-a893-4ffa-b247-51d94c7cdb1a
# ╟─ded749bf-b9fa-4e2b-b15f-0693d820a9c3
# ╠═aa1dfa87-a844-4f82-9a30-008f15f88112
# ╠═0608c93d-2f82-470c-8d9f-ca79af6b2612
# ╠═0ba5f9c8-5677-40e9-811b-25546e0df207
# ╟─acf1b36c-0412-452c-ab4d-a388f84fd1fb
# ╠═159cecd9-de77-4586-9479-383661bf3397
# ╠═d9935cc8-ec24-47e9-b39a-92c21377a161
# ╟─5714c84f-1653-4c4a-a2e4-003d8560484a
# ╠═69cf84e2-0aba-4595-8cb0-c082dbccdbe2
# ╠═9d96ede3-533e-42f7-ada1-6e71980bc6c2
# ╠═6ff92fca-6d66-4f27-8e09-11a3887e66ba
# ╠═2782903e-1d2e-47de-9109-acff4595de42
# ╟─a17e5acd-d78d-4fab-9ab2-f01bd888339d
# ╟─a3387a7e-436c-4724-aa29-92e78ea3a89f
# ╠═7e777dba-b389-4549-a93a-9b0394646c57
# ╟─f380a361-2960-471b-b29a-3bd1fe06252b
# ╟─6d2d24da-9f3f-43df-9243-fc17f85e0b01
# ╠═4f43373d-42ee-4269-9862-f53695351ea3
# ╠═ada800ba-25e2-4544-a297-c42d8b36a9ff
# ╟─7bea0c90-077f-4eca-b880-02d1289244f3
# ╟─4ea14063-99ca-4caf-a862-fbf9590c68a2
# ╠═299116ea-66f3-4e52-ab0f-594249b9dd23
# ╟─af8acfdf-32bd-43c1-82d0-99008ee4cb3e
# ╟─3738f45d-38e5-415f-a4e6-f8922df84d09
# ╠═36c1f5c8-ac43-41ea-9100-8f85c1ee3708
# ╟─8dfd4734-5648-42f2-b93f-be304b4b1f27
# ╟─77cda37c-c560-42d8-85b1-7f026d09edfe
# ╠═1fe7084a-9cf9-48a4-9e60-b712365eded9
# ╠═c2710a60-ebe1-4d01-b6d1-0d6fe45723f9
# ╟─fa610de0-f8c7-4c48-88d8-f5398ea75ae2
# ╠═d695b118-6d0d-401d-990f-85ba467cc53e
# ╠═6edc243e-59ac-4c6f-b507-80d3ec13bc21
# ╠═3bdde6cf-3b68-46d3-bf76-d68c20b661e9
# ╠═7905f581-1593-4e06-8aaf-faec05c3b306
# ╠═69b96fc3-dc9c-44de-bc7f-12bb8aba85d1
# ╟─1cdb6679-c18f-46f7-8f23-9ed6e138a7a9
# ╠═89cc730e-ab66-4f87-827c-87539ac1f54a
# ╠═045b8be8-58c6-497b-baac-8af41de76b1e
# ╟─eef918a9-b8af-4d41-85b1-bebf1c7889cc
# ╟─fd75ff49-b5de-48dc-ae89-06bf855d81b2
# ╠═d339a276-296a-4378-82ae-fe498e9b5181
# ╟─58e626f1-32fb-465a-839e-1f413411c6f3
# ╟─5ec88a5a-71e2-40c1-9913-98ced174341a
# ╠═d2c516c0-f5e5-4476-b7d6-89862f6f2472
# ╠═90f34d85-3fdc-4e2a-ada4-085154103c6b
# ╟─eac4fb9d-126b-4ba8-9078-105638416de2
# ╟─14b329fb-8053-4148-8d24-4458e592e7e3
# ╠═eebf5285-2336-4c07-a4fd-b1fd841dee52
# ╟─9ed07108-2ed0-430f-ab97-6f51297c5361
# ╠═d686c8ca-cb29-4f7c-8872-a907173b156c
# ╟─14fa89f8-d034-4286-bdb2-2c11190e17d0
# ╟─26466a2e-5554-407f-8729-e2b841f10a7e
# ╟─1189ddd5-9295-4e1c-a50e-11efed56d35b
# ╟─9e33cb14-44b6-4335-a899-7d51a9829346
# ╠═11098780-5235-40a1-9477-091ce68420a9
# ╟─4f8d8553-dfaa-4d49-a5b9-c0993fb03e85
# ╟─3468f550-da21-4e80-a030-0ab74439c1ee
# ╠═fa0bac24-e5c5-425b-b719-a0b98dead6b2
# ╠═e2038762-570f-461b-8f39-0549446d1e5b
# ╠═4277a0fb-3746-424c-80af-02f6862c3258
# ╠═00cf8e83-4953-4487-814c-7c6cd11b5e11
# ╠═f3bd26ee-ea95-4590-aa7c-da5f293b2d77
# ╟─f8ff4eca-1d38-45dc-9d85-5e0de774aa4a
# ╟─91636720-2e59-47ab-87ff-a50e9d817d18
# ╠═015bdde5-77c9-49d7-a8ba-c9c11f0033a9
# ╠═2cd21c3b-c826-40d6-9bdf-5e808c23a7d3
# ╠═c71c562b-bb4f-4688-a451-daf2040ede62
# ╟─efe707a1-41c7-439d-9e57-dfea374f355d
# ╠═62a4c01b-434b-4a2b-a46b-78bee5136dad
# ╠═07314336-5096-4fd3-a94a-70e41a7d3a6a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
