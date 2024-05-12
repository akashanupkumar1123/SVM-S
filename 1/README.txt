Why SVM ???

We have couple of other classifiers there, so why do we have to choose SVM over any other ??

Well! It does a pretty good job at classification than others. for example observe the below image.



This is a high level view of what SVM does, The yellow dashed line is the line which separates the data (we call this line ‘Decision Boundary’ (Hyperplane) in SVM), The other two lines (also Hyperplanes) help us make the right decision boundary.


The answer is “a line in more that 3 dimensions” ( in 1-D it’s called a point, in 2-D it’s called a line, in 3-D it’s called a plane, more than 3 - Hyperplane).

How is SVM’s hyperplane different from linear classifiers?

Motivation: Maximize margin: we want to find the classifier whose decision boundary is furthest away from any data point.We can express the separating hyper-plane in terms of the data points that are closest to the boundary. And these points are called

Margin is the distance between the left hyperplane and right hyperplane.



Vector , its a n dimensional object which has magnitude(length) and direction, it starts from origin(0,0).



If we work on a bit we get this ‖x‖‖y‖cos(θ)=x1y1+x2y2

∥x∥∥y∥cos(θ)=x⋅y

The orthogonal projection of a vector.


The line equation and hyperplane equation — same, its a different way to express the same thing,

It is easier to work on more than two dimensions with the hyperplane notation.

so now we know how to draw a hyperplane with the given dataset




We have a data-set , we want to draw a hyper plane something like above (which separates the data well).

How can we find the optimal hyperplane(yellow line) ?

if we maximize the margin(distance) between two hyperplanes then divide by 2 we get the decision boundary.

how do we maximize the margin??

lets take only 2 dimensions, we get the equation for hyper line is

w.x+b=0 which is same as w.x =0 (which has more dimensions)



if w.x+b=0 then we get the decision boundary

→The yellow dashed line

if w.x+b=1 then we get (+)class hyperplane

for all positive(x) points satisfy this rule (w.x+b ≥1)

if w.x+b=-1 then we get (-)class hyperplane

for all negative(x) points satisfy this rule (w.x+b≤-1)

Observe this picture.



so either we save the w and b values and keep going or we adjust the parameter (w and b) and keep going. Another optimization problem, SVM.

Adjusting parameters? Sounds like Gradient descent right? Yes!!!

It is a convex optimization problem which surely gives us global minmum value.

Once its optimized we are done!




 support vectors.

We would like to learn the weights that maximize the margin. So we have the hyperplane!