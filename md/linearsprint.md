# ML Sprint: March 17
### Linear Regression with Student GPAs

Ciao y'all! This week, we were given a small dataset and tasked to write a model in under 2 hours. We had to find correlation, if any, between student library usage and their GPA. Our data consisted of 9 groups, ranging from **under 2.00** to **above 3.74**. This is complimented by the ratio of students that utilize the library in that group.

| GPA       | Ratio | Unused | Used |
|-----------|-------|--------|------|
| >3.74     | 67%   | 5368   | 11023|
| 3.50-3.74 | 66%   | 5403   | 9674 |
| 3.25-3.49 | 61%   | 5972   | 9474 |
| 3.00-3.24 | 56%   | 7019   | 9062 |
| 2.75-2.99 | 53%   | 5555   | 6260 |
| 2.50-2.74 | 49%   | 5577   | 5374 |
| 2.25-2.49 | 45%   | 4201   | 3489 |
| 2.00-2.24 | 42%   | 3653   | 2666 |
| <2.00     | 34%   | 18059  | 4126 |

You've probably already noticed that the ratio of students using the library trickles down as the GPA lowers. But how can we find the line between these points? This is where we can apply <u>Linear Regression</u>. You can think of linear regression as a way to find the line of *best fit*--gradually adjusting a line to get closer to every point equally. 

So first of all, what are we looking for? What is the *X* and *Y* of our graph? Generally, we want to see how the ratio of library users effects the GPA bracket. We see can see how the % trends downwards as the GPA approaches **2.00**. Our data is split into bins of **.25** GPA, and the generalized extremes for **over 3.74** and **under 2.00**. Knowing this, we can choose the GPA as our *X* axis and the ratio of library users as our *Y* value.

Before delving too far, it's important to note that we will lose accuracy when a student approaches the ends of our data. I mentioned earlier that the data has <u>generalized extremes</u>, since we have **>** and **<**. For the higher end, it shouldn't really matter as **4.00** GPA is the upper limit in most cases. However, **2.00** and below are all grouped into one: we won't have as accurate of a model in the lower limits.

Now, how should we implement this?

### Programming with Libraries

Since 2 hours was the time limit, we're going to use premade libraries for this. You can make it with pure math, but it's much more time consuming. I chose to write the model using Rust. The library <u>dfdx</u> was used for modeling.

First lets get our data into Rust. Since we were given GPA ranges, we'll have to compromise on how we enter our data. I chose to just get the median GPA in a given range, and use that for the *X-component*. The *Y-component* is just the precentage. We're going to use Linear for our module.

```rust
use dfdx::{
    nn::builders::*,
    optim::{Optimizer, Sgd, SgdConfig},
    tensor::{Trace, TensorFrom},
    tensor_ops::{Backward, MeanTo},
};
use rand::prelude::*;

#[cfg(not(feature = "cuda"))]
type Device = dfdx::tensor::Cpu;

#[cfg(feature = "cuda")]
type Device = dfdx::tensor::Cuda;

type Model = Linear<1, 1>;

let (x_data, y_data): ([[f32; 1]; 9], [[f32; 1]; 9]) = (
    [[1.875], [2.125], [2.375], [2.625], [2.875], [3.125], [3.375], [3.625], [3.875]], // X
    [[0.34], [0.42], [0.45], [0.49], [0.53], [0.56], [0.61], [0.66], [0.67]] // Y
);

fn main() {
    let device = Device::default();
    let mut model = device.build_module::<Model, f32>();

    /* ... */
```

Now we have to consider how we are evaluating our model. To do this we establish our losses. I've decided to use the Mean Squared Error, but there are several ways to calculate losses to fit any niche. The Mean Squared Error, as the name suggests, is finding the average distance our predicted point is from the real point (the error). Mathematically, it would look something along the lines of:

$$
\frac{1}{n} \sum (y_r - y_p)^2
$$

where $y_r$ is our real point and $y_p$ is our prediction. dfdx provides a function for Mean Squared Error, but for the sake of verbosity we'll  *make it ourselves* .

We know how wrong our model is, so how do we correct it? This is done through an optimizer. The optimizer takes our losses and provides us with new parameters (parameters being some part of the model, like the slope of a line). For this issue, I'm going to use Stochastic Gradient Descent. The Stochastic Gradient Descent has a pretty complex definition, so lets start with the math. For our case (momentum and weight decay is not present):

$$
\theta _t \leftarrow \theta _{t-1} - \gamma \nabla f(\theta _{t-1})
$$

That's a lot of symbols! Here's what I understand. First, $\theta$ is a parameter and $t$ is an iteration. So the new iteration of a parameter is $\theta _t$. We also have $\theta _{t-1}$ which is the previous iteration of said parameter. Thus, $\theta _t \leftarrow \theta _{t-1} - x$ is telling us to minimize our parameter $\theta$ (like slope or intercept of a line) by $x$. The big one is $\nabla f(\theta _{t-1})$, the <u>gradient</u> of a function $f$. This is effectively asking us to find the largest difference in our function. This way we can adjust to the largest error, and slowly work down to the smaller errors. Lastly, there's $\gamma$, our learning rate. This is a number that determines how fast we want the model to change. If we change too fast, we might overshoot, and too slow will make learning take forever.

To sum up what the math means, we are changing a parameter $\theta _t$ by its previous iteration gradually in the direction of its largest error. This is way more complicated than Mean Squared Error for implementation, so we'll use dfdx's built in SGD optimizer.

```rust
    /* ... */

    let mut gradients = model.alloc_grads();

    let mut random = rand::thread_rng();

    let mut optimizer = Sgd::new(
        &model,
        SgdConfig {
            lr: 1e-2,
            momentum: None,
            weight_decay: None,
        },
    );

    /* ... */
```

We have all the structure set up, time to start learning. The amount of generations is utlimately up to you, I chose to do 100,000. dfdx makes the rest of the process very simple for iteration--its ergonomics are similar to that of the PyTorch library. First, we're going to sample our data since we have a Linear<1, 1> struct (as in: one in, one out). So we'll make a tensor, the data storage, through a random element in our dataset:

```rust
    /* ... */

    for _ in 0..100000 {
        let sample: usize = random.gen_range(0..8);
        let x = device.tensor(x_data[sample]);
        let y = device.tensor(y_data[sample]);

        /* ... */
```

This leads into our predicition, which at first should be pretty bad. This prediction can be turned into the Mean Squared Error with our prior formula. In Rust this would be:

```rust
        /* ... */

        let prediction = model.forward_mut(x.trace(gradients));
        println!("x: {}, real: {}, predi: {}", x[[0]], y[[0]], prediction[[0]]);
        let loss = (prediction - y.clone()).square().mean();

        /* ... */
```

Lastly, we can use this loss to calculate our gradients and optimize our model. dfdx provides a *.backwards()* function that we can utilize for this. The optimizer updates the model with these gradients, and then we reset.

```rust
        /* ... */

        gradients = loss.backward();
        optimizer.update(&mut model, &gradients).expect("Paramters missing");
        model.zero_grads(&mut gradients);
    }

    /* ... */
```

Brava! The learning segment is complete. We can test it out by making a custom tensor and predicting!

```rust
    /* ... */

    let test = device.tensor([2.56]);
    let predicition = model.forward(test);
    println!("{}", predicition[[0]]);

    /* ... */
```

This doesn't necessarily tell us if there is a correlation, but we know from observation that this value should be between 49% to 45%. Running the program gives us a 47.47% ratio for library users! But we should probably graph this, so lets export it.

```rust
    /* ... */

    model.save("gpa.npz").expect("Couldn't save model!");
}
```

*.save("")* allows us to use the model in numpy, so let's use that to graph.
