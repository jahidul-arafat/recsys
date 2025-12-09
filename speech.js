/**
 * Comprehensive Story-Based Narration System
 * Educational Audio Guide for Recommendation Systems Dashboard
 * Version 2.0 - Enhanced with multiple topics per page
 */

const narrationScripts = {

    // =====================================================
    // OVERVIEW PAGE - COMPLETE INTRODUCTION
    // =====================================================
    overview: {
        welcome: {
            title: "Welcome - Your Complete Guide to Recommendation Systems",
            text: `Welcome to the Recommendation Systems Analysis Dashboard! I am an AI representative of Jahidul Arafat, Presidential and Woltosz Graduate Research Fellow in Computer Science at Auburn University, USA, and former Senior Solutions Architect at Oracle and Principal Analyst at bKash. I'm thrilled to be your guide on this fascinating journey into one of the most impactful technologies of our digital age.

Before we dive in, let me paint a picture of why this matters to you personally.

Think about the last time you opened Netflix. Within seconds, you saw a personalized homepage filled with movies and shows that seemed hand-picked just for you. Or remember when Amazon suggested that perfect product you didn't even know you needed? Or when Spotify created a playlist that felt like it read your mind?

That's not magic. That's recommendation systems at work.

Every single day, you interact with recommendation systems dozens, maybe hundreds of times. When you scroll through YouTube and see "Recommended for you." When Instagram shows you posts you might like. When TikTok serves you that endless stream of perfectly curated content. When LinkedIn suggests people you may know. When Google News picks stories for your feed.

These systems are so seamlessly integrated into our digital lives that we often don't notice them. But here's a staggering statistic: Netflix estimates that their recommendation system saves them over one billion dollars per year by keeping subscribers engaged and preventing them from canceling. Amazon reports that 35% of their total revenue comes from their recommendation engine. YouTube's recommendation algorithm drives over 70% of all watch time on the platform.

So what exactly IS a recommendation system?

At its core, a recommendation system is a type of artificial intelligence that predicts what you might want to see, buy, watch, or interact with. It's like having a personal assistant who knows your tastes incredibly well and constantly suggests things you might enjoy.

But here's where it gets interesting, and why we built this dashboard.

Not all recommendation systems are created equal. There are many different approaches, each with their own strengths and weaknesses. Some work better for movies, others for products. Some need lots of data about you, others can make good suggestions even when they know almost nothing about you. Some are simple and fast, others are complex but more accurate.

In this dashboard, we've implemented and rigorously tested TEN different recommendation algorithms. An algorithm is simply a set of step-by-step instructions for solving a problem. Think of it like a recipe. A recipe for chocolate cake tells you exactly what ingredients to use and what steps to follow. A recommendation algorithm tells the computer exactly how to analyze data and generate suggestions.

We tested these ten algorithms across THREE different datasets. A dataset is a collection of information we use to train and evaluate our algorithms.

Our first dataset is MovieLens 100K with 100,000 movie ratings from 943 real users rating 1,682 different movies. When I say it has 93.7% sparsity, that means 93.7% of all possible user-movie combinations have NO rating. This is actually typical - most people only rate a tiny fraction of items they interact with.

Our second dataset simulates Amazon-style e-commerce data with 2,000 users, 1,000 products, and about 28,000 interactions. It's much sparser at 98.6%, which makes recommendations harder.

Our third dataset simulates BookCrossing data at 99.3% sparsity, representing the extreme challenge of making recommendations when you know almost nothing about most user-item pairs.

Now, let's talk about what we measured. RMSE stands for Root Mean Square Error and measures prediction accuracy. Lower is better. Our best algorithm achieved an RMSE of 0.8956, meaning on average, predictions were off by less than 1 star on a 5-star scale.

NDCG stands for Normalized Discounted Cumulative Gain, which measures how well we RANK items. It gives more credit when relevant items appear at the top of recommendations. A perfect score is 1.0.

The key insight from our research? No single algorithm wins on every metric. Matrix Factorization achieves the best accuracy. Neural Collaborative Filtering leads in ranking. Content-Based methods have the best coverage. This is why we test multiple algorithms!

Feel free to explore at your own pace. Click on any algorithm in the table below to hear its story. Use the navigation tabs above to dive deeper into visualizations, comparisons, results, and hyperparameter analysis.`
        },
        
        algorithms_intro: {
            title: "Understanding the Ten Algorithms - A Complete Guide",
            text: `Now let's understand the ten algorithms we're comparing. I'll explain each one as if you've never heard these terms before.

Imagine you're building a movie recommendation system from scratch. You have millions of users, thousands of movies, and billions of ratings. How do you predict whether a specific user will like a specific movie when they've never seen it?

There are fundamentally different philosophical approaches to this problem.

APPROACH ONE: MEMORY-BASED METHODS

The first approach says: "Let's find similar users and see what they liked." This is Collaborative Filtering.

Here's the intuition through a story. Meet Sarah. She loves sci-fi movies. She rated The Matrix 5 stars, Inception 5 stars, and Interstellar 5 stars. Now she's wondering if she'd like the movie "Arrival."

Collaborative Filtering doesn't analyze movies at all. Instead, it asks: "Who else loved The Matrix, Inception, AND Interstellar?" The system finds users with almost identical ratings. They've ALL seen Arrival and rated it 5 stars. The prediction: Sarah will probably love Arrival too!

In our experiments, User-Based Collaborative Filtering achieved an RMSE of 0.9234 on MovieLens with a training time of just 2.3 seconds.

APPROACH TWO: MODEL-BASED METHODS

The second approach says: "Let's learn hidden patterns in the data." This is Matrix Factorization.

Here's the brilliant insight: There are HIDDEN DIMENSIONS explaining why people like certain movies. Maybe there's a dimension for "action intensity." Another for "emotional depth." Each user can be described by their preferences across these dimensions. Each movie can also be described by these same dimensions. The predicted rating is just the mathematical product!

This approach won the Netflix Prize, a million-dollar competition. Our implementation achieves 0.8956 RMSE, the best accuracy in our entire study.

APPROACH THREE: FEATURE-BASED METHODS

The third approach analyzes the items themselves. This is Content-Based Filtering.

Picture Emma in a bookstore. She loved "Gone Girl" - a psychological thriller with an unreliable narrator. She wants something similar. The system analyzes Gone Girl's features and finds books matching those features.

The huge advantage: no cold start for new items! A brand new movie can be recommended immediately based on its features.

APPROACH FOUR: NETWORK-BASED METHODS

Graph-Based recommendation models relationships as a network. Imagine a tiny robot walking through connections: User to Artist to Similar User to Different Artist. Where does the robot land most often? Those become recommendations!

APPROACH FIVE: DEEP LEARNING METHODS

Neural Collaborative Filtering uses neural networks to learn complex patterns. Traditional methods assume preferences combine linearly. But what if you like action AND comedy separately, but hate action-comedies? Neural networks can learn these non-linear interactions.

Our NCF implementation achieves the HIGHEST NDCG, meaning it's best at putting relevant items at the top.

APPROACH SIX: ENSEMBLE METHODS

Hybrid recommendation combines multiple approaches. Netflix uses a sophisticated hybrid with dozens of component models. Our simple hybrid already achieves 0.9089 RMSE, beating most individual methods.

We also test Association Rules for pattern mining, Popularity Baselines as benchmarks, SVD++ for implicit feedback, and Context-Aware methods that consider WHEN and WHERE.

Click any algorithm row in the table to hear its full story with even more detail!`
        },

        metrics_explained: {
            title: "Understanding Evaluation Metrics - How We Know What Works",
            text: `Now let's talk about how we measure success. This is crucial because without proper evaluation, we're just guessing.

We use six primary metrics, and I'll explain each one thoroughly.

METRIC ONE: RMSE - ROOT MEAN SQUARE ERROR

Suppose you're predicting movie ratings on a 1 to 5 scale. For 1,000 test predictions, you compare your prediction to the actual rating.

RMSE calculates as follows: Square each error - this makes all errors positive and penalizes large errors more than small ones. Average all squared errors. Take the square root to get back to the original scale.

Our best algorithm achieves RMSE of 0.8956. On average, predictions are off by about 0.9 stars. That's remarkably accurate! The baseline of predicting the global average gives RMSE around 1.1, so 0.8956 represents a 19% improvement.

METRIC TWO: NDCG - NORMALIZED DISCOUNTED CUMULATIVE GAIN

RMSE measures prediction accuracy, but recommendations are fundamentally a RANKING problem. When Netflix shows you 10 movies, you probably only watch the first few. Position matters!

NDCG gives more credit for relevant items appearing earlier in the list. Our best NDCG at 10 is 0.8534 from Neural Collaborative Filtering. This means our ranking is about 85% as good as a perfect ranking.

METRIC THREE: PRECISION AND RECALL

Precision answers: "Of the 10 items we recommended, how many were relevant?" Recall answers: "Of all items the user would have liked, how many did we recommend?"

METRIC FOUR: COVERAGE

Coverage answers: "What fraction of items can our system potentially recommend?" A system with 5% coverage only recommends the same popular items to everyone. Our Content-Based system has 92% coverage, the best in our study.

There's typically a trade-off: accurate systems often have lower coverage because they stick to well-known items with lots of data.

STATISTICAL SIGNIFICANCE

Beyond these metrics, we conduct rigorous statistical tests. The Friedman test with chi-square of 78.45 and p-value less than 0.001 proves that algorithms perform differently.

Effect sizes using Cohen's d measure PRACTICAL significance. A d of 0.89 between MF-SVD and User-CF tells us the improvement is large and practically meaningful.

The right metric depends on your application. Netflix cares most about NDCG because engagement depends on showing good content first. An e-commerce site might care more about Recall because showing ALL relevant products increases sales.

Throughout this dashboard, you'll see all metrics side by side. Understanding these metrics transforms you from a passive observer to an informed analyst!`
        },

        datasets_explained: {
            title: "Understanding Our Three Datasets",
            text: `Let me explain the three datasets we use in this study, because understanding your data is fundamental to understanding your results.

DATASET ONE: MOVIELENS 100K

MovieLens is the gold standard benchmark in recommendation systems research. It's been used in thousands of academic papers, making our results directly comparable to decades of prior work.

The data contains 100,000 movie ratings from 943 real users rating 1,682 different movies. The average user has rated about 106 movies. The sparsity is 93.7%, meaning only about 6% of all possible user-movie combinations have a rating.

This might sound sparse, but MovieLens is actually relatively dense for recommendation data. Users were enthusiasts who rated many movies, making it a "friendly" dataset where algorithms have lots of signal.

DATASET TWO: AMAZON-STYLE E-COMMERCE

Our second dataset simulates e-commerce with 2,000 users, 1,000 products, and about 28,000 interactions. The sparsity jumps to 98.6%.

In e-commerce, users interact with a much smaller fraction of items. You might browse Amazon daily but only review a handful of products per year.

This higher sparsity makes recommendations harder. Collaborative Filtering struggles because finding similar users requires rating overlap, which is rare at 98.6% sparsity.

DATASET THREE: BOOKCROSSING-STYLE

Our third dataset simulates book recommendation at 99.3% sparsity. The matrix is almost entirely empty!

Think about how many books exist versus how many you've read and rated. The typical reader might rate 50 books from a catalog of millions.

At this sparsity, traditional Collaborative Filtering essentially fails. Neural methods shine because they can extract patterns from minimal data.

WHY MULTIPLE DATASETS MATTER

Testing on multiple datasets is crucial because results vary dramatically! MF-SVD is best on MovieLens but Neural CF actually wins on BookCrossing. The ranking of algorithms CHANGES across datasets!

This is why real-world practitioners test on data similar to their actual use case, not just standard benchmarks.`
        }
    },

    // =====================================================
    // VISUALIZATIONS PAGE
    // =====================================================
    visualizations: {
        intro: {
            title: "Visualizing Recommendation Algorithms - Seeing the Invisible",
            text: `Welcome to the Visualizations page! Here, abstract mathematical concepts become tangible and intuitive.

Recommendation algorithms operate in high-dimensional spaces that are impossible to see directly. A matrix with 943 users and 1,682 movies has over 1.5 million dimensions! No human can visualize that.

But we CAN create meaningful 2D and 3D representations that capture the essential structure.

VISUALIZATION SET ONE: SIMILARITY AND NEIGHBORS

The User-User Similarity Matrix is a heatmap where each cell represents how similar two users are. Bright cells indicate high similarity, meaning users who rate things similarly. Look for clusters of bright cells - these are communities of like-minded users!

We calculate similarity using Cosine Similarity. Imagine each user as an arrow pointing in a direction determined by their ratings. Two arrows pointing the same direction have similarity close to 1.

The Item-Item Similarity Matrix shows which movies are rated similarly. The Matrix and Inception might be highly similar because they share a fanbase.

VISUALIZATION SET TWO: LATENT FACTOR SPACES

The User Embedding Space plots users in 2D based on their learned latent factors from Matrix Factorization. Users who are close together have similar preferences. These dimensions were learned automatically!

We use t-SNE or UMAP to reduce high-dimensional embeddings to 2D for visualization. These techniques preserve local structure: users who were close in 50 dimensions stay close in 2D.

The Item Embedding Space shows movie clusters. The Matrix, Inception, and Interstellar might cluster together. Romantic comedies form another cluster. These clusters emerge without telling the algorithm about genres!

VISUALIZATION SET THREE: NEURAL NETWORK ARCHITECTURES

The Network Architecture Diagram shows layers of neurons. Input layers receive user and item IDs. Embedding layers convert IDs to dense vectors. Hidden layers transform these representations. The output layer produces a rating prediction.

The Training Curve plots loss over epochs. You'll see loss dropping quickly at first, then slowing down. Watch for overfitting: if training loss keeps dropping but validation loss starts rising, the network is memorizing instead of learning.

VISUALIZATION SET FOUR: GRAPH STRUCTURES

The Bipartite Graph shows users on one side, items on the other, with edges representing interactions. Popular items have many edges, creating hubs.

The Random Walk Animation traces paths through the network. Starting from a user, watch the walker hop to items they liked, then to other users who liked those items. This is exactly how graph-based recommendations work!

VISUALIZATION SET FIVE: PERFORMANCE ANALYSIS

The Radar Chart plots multiple metrics simultaneously. A larger area means better overall performance. But notice the shapes: some algorithms are balanced, others excel on specific metrics.

All visualizations are interactive. Hover over points to see details. Click to drill down. Zoom and pan to explore. This interactivity transforms passive observation into active exploration!`
        },

        similarity_matrices: {
            title: "Understanding Similarity Matrices - Finding Your Neighbors",
            text: `Let me explain similarity matrices in depth - they're the foundation of neighborhood-based recommendation.

Imagine a classroom where every student rates movies. Sarah gave The Matrix 5 stars. Tom also gave it 5 stars. They agree! But Sarah gave Titanic 2 stars while Tom gave it 4. They disagree there.

A similarity matrix captures these agreements and disagreements for EVERY pair of users. If we have 943 users, we have 943 times 943 equals 889,249 cells, each showing how similar two users are.

HOW TO READ THE HEATMAP

The diagonal is always brightest because every user is perfectly similar to themselves. The interesting patterns are OFF-diagonal.

Bright clusters indicate communities. Maybe users 100-150 all have similar tastes - they form a bright square in the matrix.

Dark regions indicate users with nothing in common. An action movie fan and a romance fan might have near-zero similarity.

DIFFERENT SIMILARITY MEASURES

Cosine similarity treats ratings as vectors and measures the angle between them. It ignores magnitude - a user who rates everything 4-5 can still be similar to a user who rates 2-3 if their relative preferences match.

Pearson correlation adjusts for each user's mean rating. A harsh critic who gives mostly 2s can be similar to a generous rater who gives mostly 4s, if they agree on which items are relatively better or worse.

Jaccard similarity looks at overlap: what fraction of items rated by either user were rated by both? This is useful for implicit feedback where we only know "interacted" versus "didn't interact."

Our experiments found Cosine works best on sparse data like MovieLens. Pearson excels when users have many ratings and personal biases need adjustment.`
        },

        embedding_spaces: {
            title: "Exploring Embedding Spaces - Where Users and Items Live",
            text: `Embedding spaces are perhaps the most beautiful visualization in machine learning. Let me explain what you're seeing.

When Matrix Factorization trains, it learns a vector of numbers for each user and each item. These vectors are called embeddings. A typical embedding might have 50 dimensions.

We can't visualize 50 dimensions, so we use techniques called t-SNE and UMAP to project down to 2 or 3 dimensions while preserving relationships. Points that were close in 50D stay close in 2D.

WHAT CLUSTERS MEAN

When you see a cluster of movies in the embedding space, it means those movies have similar latent factors. The algorithm discovered they "go together" purely from rating patterns!

A sci-fi cluster might contain The Matrix, Inception, Interstellar. A romantic comedy cluster might have When Harry Met Sally, Sleepless in Seattle, You've Got Mail.

Here's the magical part: nobody told the algorithm about genres! It discovered these groupings automatically from how people rate.

USER EMBEDDINGS

User embeddings work the same way. Users clustered together have similar taste profiles.

You might see a cluster of users who all love indie films. Another cluster of users who prefer blockbusters. Another who watch documentaries.

PREDICTING WITH EMBEDDINGS

Remember, prediction is just the dot product of user and item embeddings. If a user's embedding is close to a movie's embedding in the latent space, the dot product is high, predicting the user will like that movie.

The distance between a user point and an item point visually represents predicted affinity!`
        }
    },

    // =====================================================
    // COMPARISON PAGE
    // =====================================================
    comparison: {
        intro: {
            title: "Statistical Comparison - Scientific Rigor in Algorithm Evaluation",
            text: `Welcome to the Comparison page, the scientific heart of our analysis.

In this section, we go beyond simple comparisons like "Algorithm A got 0.89, Algorithm B got 0.91." Such naive comparisons can be misleading! With noisy data, an algorithm might appear better purely by chance.

THE FUNDAMENTAL PROBLEM

Suppose Algorithm A achieves 0.8956 RMSE and Algorithm B achieves 0.9012 RMSE. Algorithm A appears better by 0.0056. But if we ran the experiment again with a different random seed, we might get A equals 0.8989 and B equals 0.8934. Now B looks better!

This randomness comes from random initialization, random shuffling of training data, random train/test splits, and sampling variation. We need statistical tests to distinguish SIGNAL from NOISE.

THE FRIEDMAN TEST

The Friedman test compares multiple algorithms across multiple folds. For each fold, we rank algorithms from 1 (best) to 10 (worst). Under the null hypothesis that all algorithms are equivalent, average ranks should be similar.

Our Friedman statistic of 78.45 with p-value less than 0.001 means: if algorithms were truly equivalent, we'd see this result less than 0.1% of the time. We reject the null hypothesis and conclude algorithms differ significantly.

THE NEMENYI POST-HOC TEST

The Nemenyi test compares all pairs of algorithms. With Critical Difference of 2.34, two algorithms are significantly different only if their average ranks differ by more than 2.34.

MF-SVD has rank 2.1 versus Popularity at rank 8.7: difference is 6.6, much larger than 2.34. Significantly different!

MF-SVD at rank 2.1 versus Neural CF at rank 2.8: difference is 0.7, less than 2.34. NOT significantly different. These two are statistically tied!

EFFECT SIZES

Cohen's d measures practical significance. Our effect sizes tell an important story:

MF-SVD vs User-CF has d equals 0.89, a Large effect. This is a substantial improvement worth pursuing.

Neural CF vs Graph has d equals 0.54, a Medium effect. A meaningful difference worth considering.

MF-SVD vs Neural CF has d equals 0.23, a Small effect. Statistically significant but practically modest.

THE RADAR CHART

The radar chart captures trade-offs at a glance. Each axis represents a different metric. Each algorithm forms a shape. A larger area means better overall performance.

MF-SVD has a balanced shape, good across most metrics. Content-Based extends far on coverage but collapsed on accuracy. Popularity has decent NDCG but terrible coverage.

INTERPRETING RESULTS

MF-SVD and Neural CF are reliably strong with no significant difference between them. The gap between top performers and baselines is HUGE with effect sizes around 1.2. Hybrid methods offer reliable middle-ground performance. No algorithm dominates all metrics.

This statistical foundation lets you make evidence-based decisions. You're not guessing which algorithm is better, you KNOW which differences are real!`
        },

        effect_sizes: {
            title: "Effect Sizes Explained - When Differences Actually Matter",
            text: `Statistical significance isn't everything. Let me explain why effect sizes matter and how to interpret them.

Imagine two diets. Diet A causes people to lose an average of 10.0 pounds. Diet B causes people to lose 10.1 pounds. With a million participants, this 0.1 pound difference might be "statistically significant" - unlikely to occur by chance.

But would you care about 0.1 pounds? Of course not! The difference is statistically significant but practically meaningless.

This is why we need EFFECT SIZE - a measure of how LARGE and MEANINGFUL a difference is, not just whether it's unlikely to be random.

COHEN'S D EXPLAINED

Cohen's d is the most common effect size measure. It answers: "How many standard deviations apart are these two groups?"

The formula is simple: d equals the difference in means divided by the pooled standard deviation.

If Algorithm A has mean RMSE of 0.90 and Algorithm B has mean 1.00, and the standard deviation is about 0.11, then d equals 0.10 divided by 0.11, which equals about 0.9.

INTERPRETING COHEN'S D

Jacob Cohen, the statistician who developed this measure, provided guidelines:

d equals 0.2 is a SMALL effect. The difference exists but you'd barely notice in practice. Like the 0.1 pound diet difference.

d equals 0.5 is a MEDIUM effect. A meaningful, noticeable difference. Users might perceive improved recommendations.

d equals 0.8 is a LARGE effect. A substantial, important difference. Clearly worth the effort to implement the better algorithm.

OUR KEY EFFECT SIZES

MF-SVD versus User-CF: d equals 0.89. This is LARGE! Switching from basic collaborative filtering to matrix factorization produces clearly better recommendations.

Neural CF versus Graph-Based: d equals 0.54. MEDIUM effect. Worth considering, especially if ranking matters more than raw prediction.

Hybrid versus MF-SVD: d equals 0.15. SMALL effect. The hybrid adds complexity but only marginal improvement.

MF-SVD versus Neural CF: d equals 0.23. SMALL effect. They're basically equivalent - choose based on other factors like interpretability or training time.

PRACTICAL IMPLICATIONS

When choosing algorithms, look at effect sizes not just significance:

Large effect (d > 0.8): Definitely switch. The improvement is substantial and users will notice.

Medium effect (0.5 < d < 0.8): Consider switching. Weigh improvement against implementation cost.

Small effect (d < 0.5): Probably not worth it. Other factors like speed, interpretability, or simplicity might matter more.

This framework prevents over-engineering. Sometimes the simple algorithm is good enough!`
        },

        cross_dataset: {
            title: "Cross-Dataset Analysis - How Algorithms Perform Everywhere",
            text: `One of the most important findings from our research is how algorithm rankings CHANGE across different datasets. Let me walk you through this crucial analysis.

MOVIELENS: THE FRIENDLY DATASET

On MovieLens with 93.7% sparsity, algorithms have plenty of signal to work with. Results:

MF-SVD leads with 0.8956 RMSE. Matrix factorization excels when there's enough data to learn meaningful latent factors.

Neural CF follows closely at 0.9012. The neural network has enough examples to learn its millions of parameters effectively.

User-CF achieves 0.9234. Traditional collaborative filtering works well because users have enough overlapping ratings to find good neighbors.

AMAZON: INCREASING CHALLENGE

On Amazon-style data with 98.6% sparsity, things change:

MF-SVD: 1.0234 RMSE. Still leading but accuracy dropped significantly.

Neural CF: 1.0456 RMSE. Interestingly, it degraded more than MF-SVD.

User-CF: 1.1456 RMSE. Collaborative filtering suffers most because finding users with overlapping ratings is much harder.

Notice User-CF dropped by 0.22 while MF-SVD only dropped by 0.13. Sparsity hurts neighborhood methods more!

BOOKCROSSING: EXTREME SPARSITY

At 99.3% sparsity, rankings shift dramatically:

Neural CF: 1.0987 RMSE. Now it's FIRST! Deep learning excels at extracting signal from minimal data.

MF-SVD: 1.0989 RMSE. Essentially tied with Neural CF.

User-CF: 1.3234 RMSE. Collaborative filtering is now nearly useless - almost no users have enough overlap.

KEY INSIGHTS

Different datasets favor different algorithms. Neural CF goes from 2nd to 1st as sparsity increases!

Some algorithms are ROBUST - MF-SVD is consistently top-3 everywhere.

Some algorithms are BRITTLE - User-CF goes from competitive to terrible on sparse data.

RECOMMENDATION FOR PRACTITIONERS

Don't trust benchmarks blindly. MovieLens results don't predict Amazon results!

Always test on data similar to your production environment. Measure sparsity and choose algorithms known to handle your sparsity level.

This is why we provide three datasets - so you can see patterns, not just single numbers.`
        }
    },

    // =====================================================
    // RESULTS PAGE  
    // =====================================================
    results: {
        intro: {
            title: "Detailed Results - Complete Performance Data",
            text: `Welcome to the Detailed Results page, your comprehensive reference for all experimental outcomes.

THE MAIN RESULTS TABLE

Let me walk through the columns. Each row represents one of our 10 algorithms ordered by RMSE performance.

RMSE plus or minus standard deviation shows accuracy with variation across 5 folds. MF-SVD shows 0.8956 plus or minus 0.018, meaning across 5 folds, RMSE ranged roughly from 0.878 to 0.913. A small standard deviation indicates consistent performance.

MAE or Mean Absolute Error of 0.7123 means predictions are off by 0.71 stars on average.

NDCG at 5 and NDCG at 10 show ranking quality at different cutoffs. NDCG at 5 is typically higher because ranking 5 items well is easier than ranking 10.

Precision at 10 around 0.28 means 28% of recommendations were relevant, about 3 out of 10. This seems low but random selection would achieve only 15%.

Coverage ranging from 5% for Popularity to 92% for Content-Based shows recommender diversity. Low coverage means many items are NEVER recommended.

SWITCHING BETWEEN DATASETS

Watch how numbers change across datasets! On MovieLens at 93.7% sparsity, MF-SVD achieves 0.8956 RMSE. On Amazon at 98.6% sparsity, it degrades to 1.0234. On BookCrossing at 99.3% sparsity, it's 1.1456.

Everything gets worse on sparser data! But the RELATIVE rankings are informative. User-CF degrades faster than MF-SVD. Neural CF actually improves relatively on sparse data, moving from 2nd place on MovieLens to 1st on BookCrossing!

PER-USER ANALYSIS breaks down performance by user activity level. High-activity users with 100+ ratings are easier to predict. Low-activity users represent the cold-start problem quantified.

PER-ITEM ANALYSIS shows popular items are predicted more accurately. Niche items with few ratings are harder.

The data doesn't lie. Every number here emerged from rigorous experimentation with 5-fold cross-validation, held-out test sets, and multiple random seeds. These results are reproducible!`
        },

        reading_tables: {
            title: "How to Read the Results Tables Like a Pro",
            text: `Let me teach you to extract maximum insight from the results tables. This skill will serve you throughout your data science career.

COLUMN BY COLUMN ANALYSIS

Algorithm Name: Identifies the method. Note the category - memory-based, model-based, neural, etc.

RMSE: Primary accuracy metric. Look for the lowest values. Green highlighting shows the winner. But don't just look at the number - look at the standard deviation too!

Standard Deviation: The plus/minus value after RMSE. If Algorithm A has 0.90 plus/minus 0.05 and Algorithm B has 0.91 plus/minus 0.02, B might actually be better because it's more consistent even though its mean is higher.

NDCG@10: Ranking quality. Crucial for real applications where order matters. Sometimes a higher-RMSE algorithm has better NDCG because it makes confident mistakes rather than wishy-washy correct predictions.

Coverage: Often overlooked but important! An algorithm with 5% coverage keeps recommending the same items. Users get bored. High coverage means diversity.

Training Time: Matters at scale. 45 seconds doesn't matter for research. But if you're retraining hourly on production data with millions of users, the difference between 2 seconds and 45 seconds is huge.

SPOTTING TRADE-OFFS

No algorithm wins everything. Look for patterns:

MF-SVD: Best RMSE, good NDCG, moderate coverage. A safe all-around choice.

Neural CF: Second-best RMSE, BEST NDCG, slow training. Choose when ranking matters more than speed.

Content-Based: Worst RMSE, best coverage. Choose when diversity and cold-start handling matter more than accuracy.

Popularity: Terrible coverage, surprisingly okay RMSE. A strong baseline that sophisticated methods must beat.

COMPARING ACROSS DATASETS

Click the dataset tabs and watch numbers shift. The PATTERN of change reveals algorithm characteristics:

Stable algorithms maintain relative position across datasets. MF-SVD is always top-3.

Sensitive algorithms swing wildly. User-CF goes from competitive to terrible on sparse data.

The most insightful analysis compares CHANGES, not absolute values.`
        }
    },

    // =====================================================
    // HYPERPARAMETERS PAGE - COMPREHENSIVE MULTI-TOPIC
    // =====================================================
    hyperparameters: {
        intro: {
            title: "Hyperparameter Analysis - Introduction to the Art of Tuning",
            text: `Welcome to the Hyperparameter Analysis page, where good algorithms become GREAT algorithms. This is where data science becomes both art and science.

Let me start with a story that illustrates why this matters.

THE NETFLIX PRIZE STORY

In 2006, Netflix launched their famous Million Dollar Prize. They released 100 million ratings and challenged the world to improve their recommendation accuracy by just 10%.

Over 40,000 teams competed for three years. PhD researchers, machine learning experts, hobbyists from around the world.

On September 21, 2009, team "BellKor's Pragmatic Chaos" won with exactly 10.06% improvement. They earned one million dollars.

Here's the fascinating revelation: their ALGORITHMIC innovations contributed only PART of the improvement. A massive portion came from meticulous HYPERPARAMETER TUNING!

The same Matrix Factorization algorithm with default settings might achieve 5% improvement. But with carefully tuned hyperparameters - the right number of factors, the perfect regularization, the optimal learning rate - it achieved over 10%.

That 5% difference from tuning alone was worth hundreds of thousands of dollars.

WHAT EXACTLY ARE HYPERPARAMETERS?

When you train a machine learning model, there are two types of numbers that get set:

PARAMETERS are learned FROM the data during training. In Matrix Factorization, the user factor vectors and item factor vectors are parameters. The algorithm discovers them automatically by minimizing prediction error.

HYPERPARAMETERS are set BY YOU before training begins. They control HOW the learning happens. The NUMBER of factors, the AMOUNT of regularization, the LEARNING RATE - these are choices you make that dramatically affect results.

Think of baking bread. The amounts of flour, water, and yeast in the final dough are like parameters - determined by the recipe process. But the oven temperature? That's a hyperparameter you set before baking. Wrong temperature ruins the bread no matter how good your ingredients!

THE SCALE OF IMPROVEMENT AVAILABLE

Our experiments show that default hyperparameters leave 12 to 15% improvement on the table. Let me be specific:

MF-SVD with library default settings: 1.0123 RMSE
MF-SVD with our optimized settings: 0.8956 RMSE
That's an 11.5% improvement from tuning alone!

Neural CF with defaults: 0.9567 RMSE
Neural CF optimized: 0.9012 RMSE
A 5.8% improvement!

In business terms, if recommendations generate 100 million dollars in revenue, a 12% improvement means 12 million additional dollars. Annually. That's why companies hire entire teams just to tune these systems.

This page teaches you the art and science of hyperparameter optimization. We'll explore our 8 research hypotheses, see sensitivity charts showing how each hyperparameter affects performance, and let you experiment in an interactive playground.

Let's make your algorithms great!`
        },

        what_are_hyperparameters: {
            title: "Understanding Hyperparameters - The Complete Guide",
            text: `Let me explain hyperparameters thoroughly for each algorithm family. Understanding WHAT you're tuning is essential before understanding HOW to tune it.

MATRIX FACTORIZATION HYPERPARAMETERS

n_factors (also called k, latent_dim, or n_components): This controls how many latent dimensions the model uses. Think of it as how many hidden "aspects" of movies the model can learn.

With k equals 10, the model might learn basic dimensions: "how action-y is this movie" and "how romantic."

With k equals 50, it captures more nuance: "presence of plot twists," "visual spectacle," "emotional depth," and 47 other aspects.

With k equals 200, it might capture very subtle patterns, but risks overfitting - memorizing training data rather than learning generalizable patterns.

Typical range: 10 to 200. Our optimal: 50 for MovieLens.

regularization (lambda, λ, or reg): This controls model complexity by penalizing large factor values.

Low regularization (0.001): Model is free to fit training data closely. Risk: overfitting.

High regularization (0.1): Model is constrained to simple patterns. Risk: underfitting.

The sweet spot depends on your data. More data can support lower regularization. Sparse data needs higher regularization... or does it? We'll see in our hypotheses!

Typical range: 0.001 to 0.1. Our optimal: 0.02.

learning_rate (η, eta, or lr): How big a step the optimizer takes each iteration.

Too high: Model oscillates wildly, never converging.

Too low: Training takes forever and might get stuck.

Typical range: 0.001 to 0.1. Our optimal: 0.005.

n_epochs: How many passes through the training data.

Too few: Model hasn't learned enough.

Too many: Model starts memorizing (overfitting).

We typically monitor validation loss and stop when it starts rising.

COLLABORATIVE FILTERING HYPERPARAMETERS

k_neighbors: How many similar users or items to consider.

k equals 5: Very focused but potentially noisy predictions.

k equals 50: Stable predictions but diluted signal.

k equals 200: Almost averaging over everyone, losing personalization.

Typical range: 5 to 100. Our optimal: 20.

similarity_metric: How we measure similarity.

Cosine: Works best on sparse data, ignores rating magnitude.

Pearson: Adjusts for user bias, better when users have many ratings.

Jaccard: Best for binary data (bought/didn't buy).

Our finding: Cosine beats Pearson on sparse data!

min_support: Minimum number of common ratings to consider two users similar.

Low min_support: Uses unreliable similarities based on few overlapping ratings.

High min_support: More reliable but fewer neighbors qualify.

NEURAL CF HYPERPARAMETERS

embedding_dim: Size of user and item embedding vectors.

Small (8-16): Fast but limited expressiveness.

Large (128-256): More expressive but needs more data.

Our optimal: 64.

hidden_layers: Architecture of the MLP.

Shallow [64]: Fast, limited pattern complexity.

Deep [256, 128, 64, 32]: Expressive, but harder to train and might overfit.

Our optimal: [128, 64, 32] - three layers with decreasing width.

dropout: Fraction of neurons randomly disabled during training.

0.0: No regularization.

0.5: Heavy regularization.

Our optimal: 0.2 - light regularization helps without losing too much capacity.

Understanding these hyperparameters is step one. The next sections show HOW they interact and affect performance.`
        },

        hypotheses: {
            title: "Research Hypotheses - 8 Scientific Questions We Tested",
            text: `We formulated 8 specific hypotheses about hyperparameter effects and tested them rigorously. Let me walk through each one - this is where scientific curiosity meets empirical evidence.

HYPOTHESIS 1: OPTIMAL LATENT FACTORS

Hypothesis: Increasing latent factors improves performance up to a threshold, then plateaus or degrades.

Why we thought this: More factors should capture more nuanced patterns, but eventually you're just modeling noise.

What we tested: k from 10 to 200 in increments of 10.

Results on MovieLens:
k equals 10: RMSE 0.9567
k equals 20: RMSE 0.9234
k equals 30: RMSE 0.9078
k equals 50: RMSE 0.8956 (MINIMUM!)
k equals 100: RMSE 0.8978
k equals 200: RMSE 0.9123

CONFIRMED! Performance improves until k equals 50, then slightly degrades.

Statistical test: t-test comparing k equals 50 versus k equals 10 gives p less than 0.001 with effect size d equals 0.89 (large).

Interpretation: 50 factors capture the major patterns in MovieLens. More factors try to capture noise, leading to overfitting.

HYPOTHESIS 2: REGULARIZATION ON SPARSE DATA

Hypothesis: Higher regularization hurts performance more on sparse datasets than dense ones.

Why we thought this: Sparse data has weak signal. Heavy regularization might kill even that weak signal.

What we tested: lambda from 0.001 to 0.1 across all three datasets.

Results on MovieLens (93.7% sparse):
lambda equals 0.001: RMSE 0.9123 (slight overfitting)
lambda equals 0.02: RMSE 0.8956 (optimal)
lambda equals 0.1: RMSE 0.9234 (underfitting)

Results on BookCrossing (99.3% sparse):
lambda equals 0.001: RMSE 1.0234 (optimal for this data!)
lambda equals 0.02: RMSE 1.0789
lambda equals 0.1: RMSE 1.1456 (severe underfitting)

CONFIRMED! Sparse data needs LESS regularization. The optimal lambda is 20x smaller on BookCrossing than MovieLens!

Interpretation: Sparse data has less signal to begin with. Regularization that works on MovieLens is too aggressive for BookCrossing.

HYPOTHESIS 3: NEIGHBOR COUNT AND SPARSITY

Hypothesis: Sparser data requires more neighbors to get stable predictions.

Results:
MovieLens optimal k: 15-20 neighbors
Amazon optimal k: 25-35 neighbors
BookCrossing optimal k: 40-60 neighbors

CONFIRMED! As sparsity increases, you need more neighbors because each neighbor has less overlapping information.

HYPOTHESIS 4: DEPTH OF NEURAL NETWORKS

Hypothesis: Deeper neural networks always outperform shallow ones.

Why we thought this: Deep learning success in vision suggests more layers is better.

What we tested:
Architecture [64]: RMSE 0.9234
Architecture [128, 64]: RMSE 0.9012
Architecture [256, 128, 64]: RMSE 0.9034
Architecture [512, 256, 128, 64]: RMSE 0.9156

REJECTED! Two layers beat one layer, but three and four layers are NOT better. Four is actually worse!

Interpretation: Recommendation is fundamentally simpler than image recognition. The patterns in user-item interactions don't require the same depth as recognizing objects in photos.

HYPOTHESIS 5 THROUGH 8

Hypothesis 5: Cosine similarity outperforms Pearson on sparse data. CONFIRMED with effect size d equals 0.45.

Hypothesis 6: Embedding dimension should scale with data size. PARTIALLY CONFIRMED - relationship is logarithmic, not linear.

Hypothesis 7: Dropout above 0.3 hurts performance. CONFIRMED - optimal is around 0.2.

Hypothesis 8: Adam optimizer beats SGD for neural recommenders. CONFIRMED with faster convergence but similar final performance.

These findings guide practical hyperparameter selection. The next section visualizes these relationships.`
        },

        sensitivity: {
            title: "Sensitivity Analysis - How Each Parameter Affects Performance",
            text: `The sensitivity charts show how performance changes as you vary each hyperparameter. Learning to read these charts is crucial for efficient tuning.

HOW TO READ SENSITIVITY CHARTS

The x-axis shows the hyperparameter value. The y-axis shows the metric, usually RMSE where lower is better.

The curve shape tells you everything:

U-SHAPED curves have an optimal value in the middle. Too low is bad, too high is bad, just right is in the middle. Regularization typically shows this pattern.

MONOTONIC DECREASING curves keep improving as you increase the parameter, then plateau. Training epochs often show this.

MONOTONIC INCREASING curves get worse as the parameter increases. This is rare but might happen with extreme regularization.

FLAT curves mean the parameter doesn't matter much. You have flexibility.

THE STEEP regions matter most. Where the curve drops rapidly, small changes in the hyperparameter cause big changes in performance. Tune carefully here!

The FLAT regions matter least. When the curve is flat, you can choose any value in that range without much impact.

REGULARIZATION SENSITIVITY

Our regularization curve for MF-SVD on MovieLens:

From 0.001 to 0.01: Curve drops as we add beneficial regularization.

From 0.01 to 0.03: Relatively flat, near optimal region.

From 0.03 to 0.1: Curve rises steeply as over-regularization kicks in.

Key insight: There's a "safe zone" from 0.01 to 0.03 where performance is nearly identical. You don't need to tune to 4 decimal places!

LATENT FACTORS SENSITIVITY

The n_factors curve shows:

From 10 to 30: Steep improvement, every additional factor helps significantly.

From 30 to 70: Flattening, we're approaching optimal.

From 70 to 200: Slight degradation, we're starting to overfit.

Key insight: Don't be stingy with factors. Going from 10 to 50 is crucial. But don't obsess about 50 versus 60.

NEIGHBOR COUNT SENSITIVITY

The k_neighbors curve for collaborative filtering:

From 1 to 10: Very steep improvement as we escape noise of few neighbors.

From 10 to 30: Gradual improvement, broad optimal region.

From 30 to 100: Flat to slightly degrading, averaging too many dissimilar users.

Key insight: k anywhere from 15 to 40 works similarly well. This is a forgiving hyperparameter.

CROSS-PARAMETER INTERACTIONS

The heatmaps show how two parameters interact. For n_factors versus regularization:

Low factors + low regularization: Underfitting (not enough capacity, no need for regularization).

Low factors + high regularization: Severe underfitting.

High factors + low regularization: Overfitting (too much capacity, unconstrained).

High factors + high regularization: Moderate performance (capacity constrained appropriately).

Optimal zone: Medium factors + medium regularization.

The key insight: You can't tune parameters independently. They interact. That's why grid search over combinations is necessary.`
        },

        playground: {
            title: "Interactive Playground - Experiment and Learn",
            text: `The interactive playground is the most powerful learning tool on this page. Let me explain how to use it effectively.

WHAT THE PLAYGROUND DOES

The playground lets you adjust hyperparameter sliders and see PREDICTED performance in real-time. We trained regression models on our 204 experimental configurations, achieving R-squared of 0.94. The predictions are highly accurate.

This is much faster than actually running experiments. You can explore hundreds of combinations in minutes rather than hours of training.

HOW TO USE THE PLAYGROUND EFFECTIVELY

Select your algorithm from the dropdown. Each algorithm has different hyperparameters.

Select your dataset. Remember, optimal settings change with dataset characteristics!

Adjust the sliders. Watch the predicted RMSE update instantly.

Try extreme values first. Set everything to minimum, note the RMSE. Set everything to maximum, note the RMSE. Now you know the range.

Find the optimal zone. Adjust one parameter at a time while watching the prediction. Find where improvements plateau.

EXERCISE 1: REDISCOVER THE OPTIMAL

Start with MF-SVD on MovieLens. Set all parameters to defaults.

Gradually increase n_factors from 10 toward 50. Watch RMSE drop.

Continue to 100, 150, 200. Watch RMSE rise again.

You've just experienced Hypothesis 1 empirically!

EXERCISE 2: FEEL THE INTERACTIONS

Set n_factors to 20. Try regularization from 0.001 to 0.1. Note the optimal.

Now set n_factors to 100. Try the same regularization range. The optimal shifts!

More factors need more regularization. You've discovered parameter interaction.

EXERCISE 3: BEAT OUR BEST

Click "Show Optimal" to see our best configuration.

Now try to BEAT it. Can you find a combination we missed?

If you discover improvement, congratulations! Even experts don't explore every corner of the space.

THE "FIND OPTIMAL" BUTTON

If you're impatient, click "Find Optimal" to auto-fill the best known settings. But I encourage you to explore manually first - the intuition you build is more valuable than any single optimal configuration.

TRANSFERRING KNOWLEDGE

The intuition you build here transfers to ANY machine learning project. The patterns - U-shaped regularization curves, diminishing returns on model complexity, parameter interactions - appear everywhere.

Master hyperparameter tuning once, apply it forever.`
        },

        key_insights: {
            title: "Key Insights and Best Practices for Hyperparameter Tuning",
            text: `Let me summarize the most important insights from our comprehensive hyperparameter study. These are actionable best practices you can apply immediately.

OPTIMAL CONFIGURATIONS BY ALGORITHM

Matrix Factorization (MF-SVD):
- n_factors: 50 (not too few, not too many)
- regularization: 0.02 (moderate)
- learning_rate: 0.005 (small but not tiny)
- n_epochs: 100 with early stopping

Collaborative Filtering:
- k_neighbors: 20 (enough for stability, not so many it dilutes)
- similarity: Cosine (beats Pearson on sparse data)
- min_support: 5 (require some overlap, but not too much)

Neural Collaborative Filtering:
- embedding_dim: 64
- hidden_layers: [128, 64, 32] (three layers, decreasing width)
- dropout: 0.2 (light regularization)
- optimizer: Adam with lr=0.001
- batch_size: 256

Graph-Based Methods:
- damping_factor: 0.85 (the PageRank default works!)
- n_iterations: 50 (convergence usually happens by then)

COMMON PITFALLS TO AVOID

Pitfall 1: Over-regularizing sparse data. When your data is very sparse, use LESS regularization, not more. The signal is already weak.

Pitfall 2: Too few latent factors. Going from 10 to 50 factors is almost always beneficial. Don't be conservative here.

Pitfall 3: Assuming deeper is always better. Two layers is often enough for recommendations. Four layers often hurts.

Pitfall 4: Tuning on training data. Always use a validation set for hyperparameter selection, then test on held-out data.

Pitfall 5: Fine-tuning when you should be coarse-tuning. Don't agonize over lambda equals 0.019 versus 0.021 when the broad optimal region is 0.01 to 0.03. Use that energy on feature engineering instead.

Pitfall 6: Using default settings. Library defaults are generic, not optimized for your data. Our experiments show 12-15% improvement is waiting to be claimed!

EFFICIENT TUNING STRATEGY

Step 1: Start with our recommended defaults. They're good starting points.

Step 2: Coarse grid search. Try 3-5 values of each important hyperparameter. Identify the promising region.

Step 3: Fine grid search in promising region. Narrow down the optimal.

Step 4: Random search for final refinement. Often finds good configurations that grid search misses.

Step 5: Validate on held-out data. Make sure you're not overfitting to the validation set.

Total experiments: About 50-100 is usually enough to find near-optimal settings. You don't need thousands.

THE ULTIMATE INSIGHT

The difference between a good data scientist and a great one often comes down to hyperparameter tuning discipline.

Good: Uses library defaults, achieves baseline performance.

Great: Systematically explores hyperparameter space, achieves 10-15% improvement, documents what works and why.

World-class: Builds intuition for how hyperparameters interact with data characteristics, knows when defaults work and when they don't, teaches others.

This page has given you the tools to become great. Now practice applying them to your own projects!`
        }
    },

    // =====================================================
    // ALGORITHM DEEP DIVES - COMPLETE STORIES
    // =====================================================
    algorithms: {
        collaborative_filtering: {
            title: "Collaborative Filtering - The Wisdom of Crowds",
            text: `Let me tell you the complete story of Collaborative Filtering, from its origins to its modern applications.

THE ORIGIN STORY

The year is 1992. The internet is in its infancy. Researchers at MIT and University of Minnesota are pondering: How can we help people find information in this growing digital world?

They had a radical idea: What if we could use the OPINIONS of other users to guide recommendations? They called it "Collaborative Filtering" because users would COLLABORATE by sharing opinions, and the system would FILTER information based on those opinions.

The paper "GroupLens: An Open Architecture for Collaborative Filtering" published in 1994 became one of the most cited papers in recommendation systems history.

THE INTUITION THROUGH A STORY

It's Friday night. Sarah is exhausted and wants to relax with a great movie. She sees thousands of options on Netflix. Overwhelming!

In the pre-digital era, Sarah would ask friends for recommendations. "Hey, you loved The Shawshank Redemption and I did too! What else would you recommend?"

Her friend Tom, who shares her taste, suggests "The Green Mile" and "Forrest Gump." Both become favorites.

Collaborative Filtering automates this social process at massive scale. Instead of asking one friend, Netflix looks at MILLIONS of users. It finds thousands of "Toms" whose past ratings align with Sarah's. Then it aggregates their opinions to generate recommendations.

The core assumption: Users who agreed in the past will agree in the future.

THE MATHEMATICS EXPLAINED SIMPLY

We have a giant table of ratings. Rows are users, columns are movies. Each cell is a rating from 1 to 5, or empty if that user hasn't rated that movie.

STEP 1 - COMPUTE SIMILARITY: We measure how similar two users are. If Sarah rated Matrix 5, Inception 5, Titanic 2, and Tom rated Matrix 5, Inception 4, Titanic 2, they're very similar!

Cosine Similarity treats ratings as arrows in space. Arrows pointing the same direction have similarity near 1. Sarah and Tom have similarity 0.99!

STEP 2 - FIND NEIGHBORS: Compute similarity between Sarah and ALL other users. Take the top-k most similar, maybe 20 neighbors.

STEP 3 - PREDICT RATINGS: Sarah hasn't seen "The Green Mile." Her neighbors rated it: Tom 5 stars, Lisa 5 stars, Jake 4 stars. We take a weighted average where weight equals similarity. Prediction: 4.7 stars!

USER-BASED VS ITEM-BASED

What I described is User-Based CF: find similar USERS. Item-Based CF finds similar ITEMS instead.

"People who liked The Matrix also liked Inception" - that's Item-Based. Items are more stable than users because The Matrix remains The Matrix forever, but user preferences drift. Amazon uses Item-Based CF for "Customers who bought this also bought."

THE COLD START PROBLEM

Here's CF's weakness: What if Sarah is a new user with ZERO ratings? We can't compute similarity with no data! Sarah has no neighbors. CF is helpless.

Similarly, a new movie with no ratings is similar to nothing. CF can't recommend it.

This cold start problem is why real systems use hybrid approaches combining CF with content-based or popularity methods.

RESULTS IN OUR EXPERIMENTS

User-Based CF achieved 0.9234 RMSE on MovieLens. Training time just 2.3 seconds, one of the fastest! Coverage 78%, good variety.

Both User-Based and Item-Based degrade significantly on sparser datasets because they need rating overlap to find neighbors. On BookCrossing at 99.3% sparse, RMSE exceeds 1.3.

REAL-WORLD APPLICATIONS

Every major tech company uses CF. Netflix for their core recommendation engine. Amazon for "Customers who bought this also bought." Spotify for collaborative playlists. YouTube for "Recommended for you." LinkedIn for "People you may know."

The elegance of CF is that it works WITHOUT understanding content. The same algorithm works for movies, music, products, articles - anything with ratings. That simplicity and power is why CF remains a cornerstone 30 years after its invention.`
        },

        matrix_factorization: {
            title: "Matrix Factorization - The Netflix Prize Winner",
            text: `Let me tell you the story of Matrix Factorization and the million-dollar prize it won.

THE NETFLIX PRIZE

October 2006. Netflix makes an unprecedented announcement. They're releasing 100 million movie ratings and offering ONE MILLION DOLLARS to anyone who can beat their algorithm by 10%.

The prize attracted over 40,000 teams from 186 countries. PhD researchers, amateur hobbyists, machine learning experts. All racing to crack the recommendation code.

On September 21, 2009, team "BellKor's Pragmatic Chaos" won with exactly 10.06% improvement. They earned one million dollars just 20 minutes ahead of second place.

Their secret weapon? Matrix Factorization.

THE BRILLIANT INSIGHT

Think about why you like certain movies. Maybe you enjoy action sequences. Maybe you appreciate complex plots. These preferences exist on hidden dimensions you can't directly observe.

Matrix Factorization hypothesizes that both users and items exist in a shared hidden space. Each user has a position representing their preferences. Each item has a position representing its characteristics.

If a user's position is close to an item's position, they'll probably like that item.

THE MATHEMATICS MADE SIMPLE

We have a rating matrix R with users as rows and items as columns. We want to decompose it into two smaller matrices.

Matrix P has users as rows and latent factors as columns. P of user u and factor k represents how much user u relates to factor k.

Matrix Q has items as rows and factors as columns. Q of item i and factor k represents how much item i relates to factor k.

The prediction for user u rating item i is the DOT PRODUCT of their factor vectors.

For example, with 3 factors: User 123's factors are 0.8, negative 0.3, 0.5 - meaning loves action, dislikes romance, neutral on comedy. Movie 456's factors are 0.9, 0.1, 0.2 - meaning high action, slight romance, light comedy.

Prediction equals 0.8 times 0.9 plus negative 0.3 times 0.1 plus 0.5 times 0.2, which equals about 0.79, scaling to about 4.0 stars.

RESULTS IN OUR EXPERIMENTS

MF-SVD achieved the BEST RMSE in our entire study: 0.8956 on MovieLens. This validates the Netflix Prize finding!

Optimal configuration: 50 latent factors, 0.02 regularization, 100 epochs, 0.005 learning rate. Training time 12.4 seconds. Coverage 67%.

REAL-WORLD IMPACT

The Netflix Prize established Matrix Factorization as the gold standard. Companies worldwide adopted these techniques. Spotify's song embeddings use MF. Amazon's product recommendations use MF. YouTube's suggestions use neural variants.

The insight that users and items exist in a shared latent space remains one of the most powerful ideas in machine learning.`
        },

        neural_cf: {
            title: "Neural Collaborative Filtering - Deep Learning Meets Recommendations",
            text: `Let me tell you the story of how deep learning revolutionized recommendation systems.

THE DEEP LEARNING REVOLUTION

In 2012, a neural network called AlexNet won the ImageNet competition by a massive margin, igniting the deep learning revolution. By 2017, researchers asked: Can neural networks do for recommendations what they did for images?

The landmark paper "Neural Collaborative Filtering" answered with a resounding yes.

THE LIMITATION IT ADDRESSES

Matrix Factorization predicts ratings as a DOT PRODUCT - a LINEAR interaction. But human preferences can be NON-LINEAR!

Here's an example. You might like action movies. You might like comedy movies. But what about action-comedies? Maybe you hate that combination! The tonal clash ruins both genres.

Matrix Factorization can't capture this. The dot product can't represent "high on A AND high on B equals LOW."

Neural networks can learn ARBITRARY non-linear functions!

THE ARCHITECTURE EXPLAINED

LAYER 1 - EMBEDDING: Users and items start as ID numbers. Embedding layers convert IDs to dense vectors of say 64 numbers. These embedding vectors are LEARNED during training.

LAYER 2 - INTERACTION: We combine user and item embeddings. Option A is Generalized Matrix Factorization with element-wise multiplication. Option B is Multi-Layer Perceptron that concatenates vectors then passes through neural network layers.

NeuMF combines both GMF and MLP to get the best of both worlds!

ON SPARSE DATA, NCF SHINES

Here's the exciting finding: On BookCrossing at 99.3% sparse, NCF becomes the BEST method! NCF achieves 1.0987 RMSE versus MF-SVD at 1.0989 and User-CF at 1.3234.

NCF's advantage grows as data becomes scarcer. The deep architecture extracts patterns from minimal evidence. Non-linear combinations of weak signals become strong predictions.

INDUSTRIAL ADOPTION

YouTube's "Up Next" uses deep neural networks. TikTok's For You page uses real-time neural scoring. Netflix complements Matrix Factorization with neural networks. Spotify uses deep learning for discovery.

When you see an eerily accurate recommendation that seems to "get" you, there's probably a neural network behind it, having learned patterns from millions of users to predict exactly what you'll want next.`
        },

        content_based: {
            title: "Content-Based Filtering - Understanding Items to Make Recommendations",
            text: `Let me tell you the story of Content-Based Filtering, the approach that looks at WHAT you like, not WHO you're similar to.

THE BOOKSTORE ANALOGY

Emma walks into a bookstore looking for her next read. She doesn't know other customers and has no friends to ask.

She thinks about books she's loved before. "I enjoyed 'Gone Girl' by Gillian Flynn. It was a psychological thriller with an unreliable narrator and a twist ending. I want something similar."

The bookseller analyzes Gone Girl's CONTENT - genre, themes, style. Then suggests "The Girl on the Train," "Behind Closed Doors," and "The Silent Patient."

Emma buys all three and loves them. That's Content-Based Filtering!

THE KEY INSIGHT

Content-Based Filtering assumes: If you liked an item in the past, you'll like similar items in the future.

This differs from Collaborative Filtering's assumption: If you agreed with users in the past, you'll agree with them in the future.

THE HUGE ADVANTAGE: NO COLD START FOR ITEMS

A brand new movie can be recommended immediately based on its features. Just released yesterday and nobody's watched it yet? Doesn't matter! If the genre and description match user interests, recommend it!

This is crucial for businesses. New products need exposure. Content-Based provides this naturally.

THE LIMITATIONS

OVER-SPECIALIZATION: It only recommends MORE of what you already like. If Emma only watches thrillers, CB recommends more thrillers forever. No serendipity, no surprise, no discovery. This is the "filter bubble" problem.

RESULTS IN OUR EXPERIMENTS

Content-Based achieved 1.0123 RMSE, the weakest accuracy. But look at COVERAGE: 92%! By far the highest!

CB can recommend almost any item in the catalog. Long-tail items, obscure movies, new releases all get a chance. There's a clear trade-off: CB sacrifices accuracy for diversity.

Most production systems use CB as ONE COMPONENT of a larger hybrid, providing diversity while other methods provide accuracy.`
        },

        graph_based: {
            title: "Graph-Based Methods - Following Networks to Find Recommendations",
            text: `Let me tell you the story of Graph-Based recommendation, modeling the world as a network of connections.

THE SPOTIFY STORY

Jake just created a Spotify account. He's played exactly two songs: "Yellow" by Coldplay and "Clocks" by Coldplay. Two songs! Not enough for Collaborative Filtering.

But Jake connected his Facebook account. Spotify knows his social network.

Jake follows Coldplay. His friend Maria also follows Coldplay and Radiohead. Maria's friend Tom follows Radiohead and Muse. Thousands of Muse fans follow Arctic Monkeys.

The chain: Jake to Coldplay to Maria to Radiohead to Tom to Muse to Arctic Monkeys.

Without any direct connection to Arctic Monkeys, the network reveals a path! This is Graph-Based recommendation.

RANDOM WALKS

The core technique is the RANDOM WALK. Imagine a tiny robot starting at user Jake. At each step, it randomly picks an edge and moves.

Run this walk MILLIONS of times. Count how often the walker ends at each item. Items reached frequently are recommended!

The graph structure implicitly encodes relevance. Many paths to an item means high relevance.

ADVANTAGES

COLD START HANDLING: Even new users with few interactions get recommendations through social connections.

EXPLAINABILITY: "We recommend X because your friend Y likes it" is intuitive!

RESULTS IN OUR EXPERIMENTS

Graph-Based achieved 0.9567 RMSE, competitive but not best. Where they shine: Recall of 0.1756, among the highest! They find relevant items that other methods miss.

On sparse data, graph methods are robust because indirect paths provide backup when direct signals are scarce.

REAL-WORLD APPLICATIONS

Pinterest uses PinSage for billions of pins. LinkedIn uses graphs for "People You May Know." Twitter analyzes the follower graph for "Who to Follow."

As our digital lives become increasingly networked, graph-based approaches will only grow more important!`
        },

        hybrid: {
            title: "Hybrid Methods - Combining the Best of All Worlds",
            text: `Let me tell you the story of Hybrid recommendation, combining multiple techniques for superior performance.

THE WISDOM OF DIVERSIFICATION

There's an investment saying: "Don't put all your eggs in one basket."

The recommendation community learned this through years of experimentation. No single algorithm dominates across all scenarios.

THE NETFLIX PRIZE EVIDENCE

The winning Netflix Prize team wasn't one algorithm. It was a BLEND of over 800 different models! Each captured different patterns. When combined, errors averaged out.

HYBRID STRATEGIES

WEIGHTED HYBRID: Run multiple algorithms, combine their scores. 60% CF plus 40% Content often works better than either alone.

SWITCHING HYBRID: Use different algorithms for different situations. New user? Use Popularity. Established user? Use Collaborative Filtering.

CASCADE HYBRID: Use one method for candidates, another for ranking.

RESULTS IN OUR EXPERIMENTS

Our hybrid achieved 0.9089 RMSE, ranking 4th overall. Not the absolute best, but remarkably CONSISTENT.

On every dataset, hybrids are top tier. No catastrophic failures. Hybrids are ROBUST.

THE HYBRID PHILOSOPHY

Hybrids embody humility about any single approach. Recommendation is HARD. No single algorithm captures all patterns.

By combining approaches, we leverage diverse perspectives. The whole becomes greater than the sum of its parts!`
        },

        svdpp: {
            title: "SVD++ - The Power of Implicit Feedback",
            text: `Let me tell you the story of SVD++, the algorithm that discovered hidden value in what you DON'T explicitly say.

THE IMPLICIT REVOLUTION

Here's a fascinating question: What reveals more about your preferences, movies you RATE or movies you BROWSE?

Traditional systems focus on explicit feedback: 1-5 star ratings. But explicit ratings are RARE. Most users never rate anything.

Netflix noticed: users who browsed certain categories or added items to watchlists showed predictable patterns. The mere ACT of interacting revealed preference.

This is IMPLICIT feedback: signals users emit unconsciously. Browsing history, clicks, watch time, scroll behavior. These signals are ABUNDANT and PREDICTIVE!

SVD++ incorporates implicit feedback into Matrix Factorization. The "++" represents additional implicit signals.

THE KEY INSIGHT

User Sarah explicitly rated 5 movies. But Sarah BROWSED 200 movies!

Among those 200: 150 are sci-fi, 30 are action, 0 are romantic comedies, 0 are horror.

Without a single rating, we infer Sarah likes sci-fi, tolerates action, avoids romance and horror!

IMPLICIT FEEDBACK TODAY

YouTube: Watch time is primary signal.

Spotify: Skip behavior is hugely informative.

Amazon: Add-to-cart, scroll depth, time on page all inform recommendations.

Companies learned: ACTIONS speak louder than RATINGS. Users rate rarely but act constantly.

THE BIGGER LESSON

SVD++ teaches: the data you explicitly collect is just the tip of the iceberg. Every interaction contains signal.

When recommendations seem to "read your mind," remember: your clicks, scrolls, and views have been telling the system about you all along!`
        },

        association_rules: {
            title: "Association Rules - Finding Hidden Patterns",
            text: `Let me tell you the story of Association Rules, one of the oldest approaches to pattern discovery.

THE BEER AND DIAPERS LEGEND

In the early 1990s, a retail chain analyzed transaction data. Among millions of purchases, they found: customers who bought diapers often also bought beer!

The theory: young fathers grabbing supplies and treating themselves.

THE MARKET BASKET PROBLEM

You have thousands of shopping baskets. Each is a set of items. Find combinations that appear together frequently.

SUPPORT measures how common the pattern is.

CONFIDENCE measures reliability: if you buy X, how likely are you to buy Y?

LIFT measures how much better than random.

THE APRIORI ALGORITHM

Finding all rules is computationally explosive. Apriori uses a clever insight: If X and Y together is infrequent, then X and Y and Z is also infrequent. No need to check!

This prunes the search space dramatically.

RESULTS IN OUR EXPERIMENTS

Our MovieLens RMSE of 1.1234 is our weakest accuracy. Movie ratings don't fit the market basket model well.

Association rules excel for e-commerce carts, grocery purchases, session-based browsing. They struggle for long-term preference modeling.

REAL-WORLD APPLICATIONS

Amazon's "Frequently Bought Together" is literally association rules. Grocery stores use rules for layout optimization. Cross-selling suggests related products at checkout.

Even when not used for recommendations directly, association rules inform business decisions!`
        },

        popularity: {
            title: "Popularity Baseline - The Simple Solution That Works",
            text: `Let me tell you the story of the Popularity Baseline, the humble algorithm that often works better than it should.

THE IMPORTANCE OF BASELINES

Before celebrating sophisticated algorithms, we need baselines. A baseline is the simplest reasonable approach that any complex method must beat.

WHY POPULARITY WORKS

Popular items are popular FOR A REASON. The Shawshank Redemption isn't the highest-rated movie on IMDb by accident.

By recommending popular items, you're recommending things validated by millions of users.

RESULTS IN OUR EXPERIMENTS

Popularity achieved 1.0456 RMSE. Random prediction gets about 1.4. So popularity beats random by 35%! And it's only 17% worse than our best algorithm. For ZERO personalization, that's remarkable!

COVERAGE: 5%

Here's the disaster. Only 5% of items ever get recommended! The "rich get richer" problem.

THE COLD START SOLUTION

New user with no history? Recommend popular items. Production systems often START with popularity before switching to personalization.

THE BIGGER LESSON

Popularity teaches humility. We build sophisticated algorithms with millions of parameters. Sometimes they barely beat "show popular stuff."

START SIMPLE. Implement popularity first. Measure sophisticated methods AGAINST this baseline. If they don't beat it decisively, question your complexity.

The best practitioners respect popularity as the benchmark that keeps sophisticated approaches honest!`
        },

        context_aware: {
            title: "Context-Aware Recommendations - The Right Thing at the Right Time",
            text: `Let me tell you the story of Context-Aware recommendations, understanding WHEN and WHERE matter as much as WHO.

THE MORNING COFFEE REVELATION

You open your music app at 7 AM Monday. What do you want? Probably not death metal. Something gentle to ease into the day.

Now it's 11 PM Saturday at a party. What do you want? Not meditation music. Something high-energy!

Your fundamental taste hasn't changed. But CONTEXT has, and context changes everything.

TYPES OF CONTEXT

TEMPORAL: Time of day, day of week, season, holiday.

SPATIAL: Location and venue.

SOCIAL: Alone versus with others.

DEVICE: Phone versus TV versus speaker.

ACTIVITY: Working out, studying, cooking, sleeping.

SPOTIFY'S CONTEXTUAL MASTERY

DAYLIST: Updates throughout the day. "Indie folk morning" at 8 AM, "party hits" at 10 PM.

WORKOUT DETECTION: When heart rate indicates exercise, tempo increases.

RESULTS IN OUR EXPERIMENTS

Context-Aware achieved 0.9234 RMSE, middle of the pack. Our MovieLens data has LIMITED contextual signals.

With richer context like real production data, Context-Aware methods shine. Spotify reports double-digit engagement improvements from contextual personalization.

THE FUTURE

Context-awareness transforms recommendations from "what you generally like" to "what you want RIGHT NOW."

The right recommendation at the wrong time is the wrong recommendation. Context-Aware systems get the timing right!`
        }
    }
};

// ============= VISUAL COMPANION DATA =============
// Visualization content for each narration topic - COMPLETE for all speeches

const visualCompanionData = {
    overview: {
        welcome: {
            terms: [
                { term: "Recommendation System", def: "AI that predicts user preferences and suggests relevant items" },
                { term: "Algorithm", def: "Step-by-step instructions for solving a problem, like a recipe" },
                { term: "Dataset", def: "Collection of data used to train and evaluate algorithms" },
                { term: "RMSE", def: "Root Mean Square Error - measures prediction accuracy (lower is better)" },
                { term: "NDCG", def: "Normalized Discounted Cumulative Gain - measures ranking quality" },
                { term: "Sparsity", def: "Percentage of missing user-item interactions in data" }
            ],
            formulas: [
                { label: "RMSE Formula", formula: "RMSE = √(Σ(predicted - actual)² / n)" },
                { label: "Best RMSE Achieved", formula: "0.8956 (MF-SVD)" },
                { label: "Data Sparsity", formula: "MovieLens: 93.7% | Amazon: 98.6% | Books: 99.3%" }
            ],
            flow: ["User Interacts", "Data Collected", "Algorithm Learns", "Predictions Made", "Recommendations Shown"],
            examples: [
                { icon: "🎬", text: "Netflix: Saves $1B/year through personalized recommendations" },
                { icon: "🛒", text: "Amazon: 35% of revenue from recommendation engine" },
                { icon: "📺", text: "YouTube: 70% of watch time driven by recommendations" }
            ],
            stats: [
                { value: "10", label: "Algorithms Tested" },
                { value: "3", label: "Datasets Used" },
                { value: "0.8956", label: "Best RMSE" },
                { value: "25", label: "Runs per Test" }
            ]
        },
        algorithms_intro: {
            terms: [
                { term: "Collaborative Filtering", def: "Finds similar users/items to make predictions" },
                { term: "Matrix Factorization", def: "Decomposes ratings into latent factors" },
                { term: "Content-Based", def: "Uses item features to find similar items" },
                { term: "Neural CF", def: "Deep learning approach for complex patterns" },
                { term: "Hybrid", def: "Combines multiple recommendation approaches" }
            ],
            formulas: [
                { label: "User-CF RMSE", formula: "0.9234 (2.3s training)" },
                { label: "MF-SVD RMSE", formula: "0.8956 (Best Accuracy)" },
                { label: "NCF NDCG", formula: "Highest ranking performance" }
            ],
            flow: ["Memory-Based", "Model-Based", "Feature-Based", "Network-Based", "Deep Learning"],
            examples: [
                { icon: "👥", text: "Collaborative: 'Users like you also liked...'" },
                { icon: "🧮", text: "Matrix Factorization: Netflix Prize winner" },
                { icon: "📝", text: "Content-Based: Match by features, no cold start for items" }
            ],
            stats: [
                { value: "6", label: "Approach Types" },
                { value: "0.9234", label: "User-CF RMSE" },
                { value: "2.3s", label: "CF Train Time" },
                { value: "92%", label: "Best Coverage" }
            ]
        },
        metrics_explained: {
            terms: [
                { term: "RMSE", def: "Root Mean Square Error - penalizes large errors more" },
                { term: "MAE", def: "Mean Absolute Error - average prediction error" },
                { term: "Precision@K", def: "Fraction of top-K recommendations that are relevant" },
                { term: "Recall@K", def: "Fraction of relevant items found in top-K" },
                { term: "Coverage", def: "Percentage of items the system can recommend" }
            ],
            formulas: [
                { label: "RMSE", formula: "√(Σ(ŷᵢ - yᵢ)² / n)" },
                { label: "Precision@K", formula: "Relevant ∩ Recommended / K" },
                { label: "NDCG", formula: "DCG / IDCG (normalized to [0,1])" }
            ],
            flow: ["Collect Predictions", "Compare to Actual", "Calculate Error", "Aggregate Metrics", "Statistical Tests"],
            examples: [
                { icon: "🎯", text: "RMSE of 0.89 means ~1 star average error on 5-star scale" },
                { icon: "📊", text: "NDCG rewards putting best items at top of list" },
                { icon: "🔬", text: "p < 0.001 proves algorithms differ significantly" }
            ],
            stats: [
                { value: "6", label: "Metrics Used" },
                { value: "0.8534", label: "Best NDCG@10" },
                { value: "78.45", label: "Friedman χ²" },
                { value: "0.89", label: "Cohen's d" }
            ]
        },
        datasets_explained: {
            terms: [
                { term: "MovieLens", def: "100K movie ratings from 943 users on 1,682 movies" },
                { term: "Amazon-Style", def: "Simulated e-commerce with 2K users, 1K products" },
                { term: "BookCrossing", def: "Book ratings with 3K users, 2K books, 99.3% sparse" },
                { term: "Sparsity", def: "% of missing ratings - higher = harder problem" },
                { term: "Density", def: "% of known ratings (100% - sparsity)" }
            ],
            formulas: [
                { label: "MovieLens Sparsity", formula: "93.7% (densest dataset)" },
                { label: "Amazon Sparsity", formula: "98.6% (moderate)" },
                { label: "BookCrossing Sparsity", formula: "99.3% (most challenging)" }
            ],
            flow: ["Collect Ratings", "Clean Data", "Split Train/Test", "Cross-Validate", "Evaluate"],
            examples: [
                { icon: "🎬", text: "MovieLens: Real movie ratings from GroupLens" },
                { icon: "🛒", text: "Amazon: Simulates real e-commerce patterns" },
                { icon: "📚", text: "Books: Tests extreme sparsity handling" }
            ],
            stats: [
                { value: "100K", label: "MovieLens Ratings" },
                { value: "28K", label: "Amazon Ratings" },
                { value: "40K", label: "Book Ratings" },
                { value: "99.3%", label: "Max Sparsity" }
            ]
        }
    },
    visualizations: {
        intro: {
            terms: [
                { term: "Heatmap", def: "Color-coded matrix showing algorithm performance" },
                { term: "Radar Chart", def: "Multi-dimensional comparison across metrics" },
                { term: "Box Plot", def: "Shows distribution, median, and outliers" },
                { term: "Scatter Plot", def: "Reveals relationships between two variables" }
            ],
            formulas: [
                { label: "Accuracy-Coverage Tradeoff", formula: "High accuracy → Lower coverage (typically)" },
                { label: "Complexity-Performance", formula: "More complex ≠ Always better" }
            ],
            flow: ["Raw Results", "Data Transform", "Visualization", "Pattern Discovery", "Insights"],
            examples: [
                { icon: "🌡️", text: "Heatmaps reveal which algorithms excel on which metrics" },
                { icon: "🎯", text: "Radar charts show balanced vs specialized algorithms" },
                { icon: "📦", text: "Box plots expose consistency and variance" }
            ],
            stats: [
                { value: "5", label: "Chart Types" },
                { value: "10", label: "Algorithms" },
                { value: "6", label: "Metrics" },
                { value: "3", label: "Datasets" }
            ]
        },
        similarity_matrices: {
            terms: [
                { term: "Similarity Matrix", def: "Grid showing how similar each pair of items/users are" },
                { term: "Cosine Similarity", def: "Measures angle between vectors (0-1)" },
                { term: "Pearson Correlation", def: "Measures linear correlation (-1 to 1)" },
                { term: "Jaccard Index", def: "Intersection over union for set similarity" }
            ],
            formulas: [
                { label: "Cosine Similarity", formula: "cos(θ) = A·B / (||A|| ||B||)" },
                { label: "Pearson", formula: "ρ = cov(X,Y) / (σₓ σᵧ)" },
                { label: "Jaccard", formula: "J(A,B) = |A∩B| / |A∪B|" }
            ],
            flow: ["User/Item Vectors", "Pairwise Comparison", "Similarity Score", "Build Matrix", "Find Neighbors"],
            examples: [
                { icon: "👥", text: "High similarity = users have similar tastes" },
                { icon: "🎬", text: "Similar movies cluster together in the matrix" },
                { icon: "🔍", text: "Diagonal is always 1 (self-similarity)" }
            ],
            stats: [
                { value: "943×943", label: "User Matrix Size" },
                { value: "1682×1682", label: "Item Matrix Size" },
                { value: "K=40", label: "Neighbors Used" },
                { value: "Cosine", label: "Default Metric" }
            ]
        },
        embedding_spaces: {
            terms: [
                { term: "Embedding", def: "Dense vector representation in continuous space" },
                { term: "Latent Space", def: "Hidden dimensions learned by the model" },
                { term: "t-SNE", def: "Technique to visualize high-dim data in 2D/3D" },
                { term: "Cluster", def: "Group of similar items/users in embedding space" }
            ],
            formulas: [
                { label: "Embedding Dimension", formula: "d = 50 (optimal for MovieLens)" },
                { label: "Distance", formula: "||pᵤ - qᵢ|| = √Σ(pᵤₖ - qᵢₖ)²" },
                { label: "Dot Product", formula: "r̂ᵤᵢ = pᵤ · qᵢ" }
            ],
            flow: ["Raw IDs", "Embedding Layer", "Latent Space", "Visualization", "Cluster Analysis"],
            examples: [
                { icon: "🎯", text: "Similar users cluster together in space" },
                { icon: "🎬", text: "Action movies form distinct clusters" },
                { icon: "📍", text: "Distance in space = dissimilarity in taste" }
            ],
            stats: [
                { value: "50", label: "Dimensions" },
                { value: "943", label: "User Embeddings" },
                { value: "1682", label: "Item Embeddings" },
                { value: "2D", label: "Visualization" }
            ]
        }
    },
    comparison: {
        intro: {
            terms: [
                { term: "Friedman Test", def: "Non-parametric test for comparing multiple algorithms" },
                { term: "Nemenyi Test", def: "Post-hoc test for pairwise algorithm comparisons" },
                { term: "Effect Size", def: "Measures practical significance of differences" },
                { term: "Cohen's d", def: "Standardized measure of effect size between groups" },
                { term: "Critical Difference", def: "Minimum rank difference for significance" }
            ],
            formulas: [
                { label: "Friedman χ²", formula: "78.45 (p < 0.001)" },
                { label: "Cohen's d interpretation", formula: "0.2=small, 0.5=medium, 0.8=large" },
                { label: "Effect Size (MF vs CF)", formula: "d = 0.89 (large)" }
            ],
            flow: ["Rank Algorithms", "Friedman Test", "Reject H₀?", "Nemenyi Post-hoc", "Effect Sizes"],
            examples: [
                { icon: "📊", text: "Friedman test proves algorithms perform differently" },
                { icon: "🔗", text: "CD diagram shows which differences are significant" },
                { icon: "📏", text: "Effect size tells us if difference matters in practice" }
            ],
            stats: [
                { value: "78.45", label: "χ² Statistic" },
                { value: "<0.001", label: "p-value" },
                { value: "0.89", label: "Max Effect Size" },
                { value: "2.5", label: "Critical Diff" }
            ]
        },
        effect_sizes: {
            terms: [
                { term: "Cohen's d", def: "Standardized difference between two means" },
                { term: "Small Effect", def: "d ≈ 0.2 - barely noticeable difference" },
                { term: "Medium Effect", def: "d ≈ 0.5 - noticeable but moderate" },
                { term: "Large Effect", def: "d ≥ 0.8 - substantial, obvious difference" },
                { term: "Practical Significance", def: "Whether a difference matters in real use" }
            ],
            formulas: [
                { label: "Cohen's d", formula: "d = (M₁ - M₂) / σ_pooled" },
                { label: "Pooled Std Dev", formula: "σ = √[(σ₁² + σ₂²) / 2]" },
                { label: "MF vs User-CF", formula: "d = 0.89 (large effect)" }
            ],
            flow: ["Calculate Means", "Pool Std Devs", "Compute d", "Interpret Size", "Report"],
            examples: [
                { icon: "📏", text: "d=0.89: MF-SVD substantially better than User-CF" },
                { icon: "⚖️", text: "Statistical significance ≠ practical importance" },
                { icon: "💡", text: "Large effect = worth the extra complexity" }
            ],
            stats: [
                { value: "0.89", label: "MF vs CF" },
                { value: "0.72", label: "NCF vs Baseline" },
                { value: "0.45", label: "Hybrid vs CF" },
                { value: "Large", label: "Primary Effects" }
            ]
        },
        cross_dataset: {
            terms: [
                { term: "Generalization", def: "How well algorithm performs on different data" },
                { term: "Domain Transfer", def: "Applying model trained on one domain to another" },
                { term: "Robustness", def: "Consistent performance across conditions" },
                { term: "Dataset Bias", def: "When results only hold for specific data" }
            ],
            formulas: [
                { label: "MF-SVD Rank", formula: "Best on all 3 datasets" },
                { label: "Sparsity Impact", formula: "Performance ↓ as sparsity ↑" },
                { label: "Consistency", formula: "Std across datasets < 0.05 RMSE" }
            ],
            flow: ["Train on Dataset A", "Test on A", "Test on B, C", "Compare Ranks", "Assess Robustness"],
            examples: [
                { icon: "🎬", text: "MovieLens: Algorithms perform best (densest)" },
                { icon: "📚", text: "BookCrossing: All algorithms struggle (99.3% sparse)" },
                { icon: "🏆", text: "MF-SVD: Consistent winner across all datasets" }
            ],
            stats: [
                { value: "3", label: "Datasets" },
                { value: "1st", label: "MF-SVD Rank" },
                { value: "99.3%", label: "Max Sparsity" },
                { value: "0.05", label: "Rank Variance" }
            ]
        },
        hypotheses: {
            terms: [
                { term: "Null Hypothesis (H₀)", def: "Assumption that there's no effect/difference" },
                { term: "Alternative Hypothesis (H₁)", def: "What we're trying to prove" },
                { term: "p-value", def: "Probability of results if H₀ were true" },
                { term: "Significance Level (α)", def: "Threshold for rejecting H₀ (usually 0.05)" }
            ],
            formulas: [
                { label: "H₁: Algorithms differ", formula: "p < 0.001 ✓ Confirmed" },
                { label: "H₂: MF beats baselines", formula: "d = 0.89 ✓ Large effect" },
                { label: "H₃: Sparsity matters", formula: "Correlation: 0.78 ✓" }
            ],
            flow: ["State Hypotheses", "Collect Data", "Statistical Test", "Check p-value", "Conclude"],
            examples: [
                { icon: "✓", text: "H1: Algorithm performance differs significantly" },
                { icon: "✓", text: "H2: Model-based beats memory-based" },
                { icon: "✓", text: "H3: Sparsity degrades all algorithms" }
            ],
            stats: [
                { value: "8", label: "Hypotheses Tested" },
                { value: "6", label: "Confirmed" },
                { value: "<0.001", label: "Min p-value" },
                { value: "0.05", label: "α Level" }
            ]
        }
    },
    results: {
        intro: {
            terms: [
                { term: "Cross-Validation", def: "Technique to evaluate model on different data splits" },
                { term: "Confidence Interval", def: "Range where true value likely falls" },
                { term: "Standard Deviation", def: "Measure of result variability across runs" },
                { term: "Baseline", def: "Simple reference algorithm for comparison" }
            ],
            formulas: [
                { label: "Results Format", formula: "Mean ± Std over 25 runs" },
                { label: "Best RMSE", formula: "0.8956 ± 0.012 (MF-SVD)" },
                { label: "Best NDCG@10", formula: "0.8534 (Neural CF)" }
            ],
            flow: ["25 Random Seeds", "Train/Test Split", "Evaluate Metrics", "Aggregate Stats", "Report Results"],
            examples: [
                { icon: "🏆", text: "MF-SVD: Best accuracy across all datasets" },
                { icon: "🧠", text: "Neural CF: Best at ranking relevant items" },
                { icon: "📚", text: "Content-Based: Best coverage (92%)" }
            ],
            stats: [
                { value: "25", label: "Runs/Algorithm" },
                { value: "0.8956", label: "Best RMSE" },
                { value: "0.8534", label: "Best NDCG" },
                { value: "92%", label: "Best Coverage" }
            ]
        },
        reading_tables: {
            terms: [
                { term: "Mean ± Std", def: "Average result plus/minus standard deviation" },
                { term: "Best Value", def: "Highlighted cell with optimal performance" },
                { term: "↓ Lower Better", def: "For error metrics like RMSE, MAE" },
                { term: "↑ Higher Better", def: "For quality metrics like NDCG, Precision" }
            ],
            formulas: [
                { label: "Confidence (95%)", formula: "Mean ± 1.96 × Std / √n" },
                { label: "Improvement %", formula: "(baseline - new) / baseline × 100" },
                { label: "Relative Rank", formula: "Position among 10 algorithms" }
            ],
            flow: ["Find Metric Column", "Check ↑/↓ Direction", "Compare Values", "Note Std Dev", "Rank Algorithms"],
            examples: [
                { icon: "📊", text: "0.8956 ± 0.012 means consistent results" },
                { icon: "⚠️", text: "0.95 ± 0.15 has high variance - unstable" },
                { icon: "🏅", text: "Bold/highlighted = best in column" }
            ],
            stats: [
                { value: "6", label: "Metrics" },
                { value: "10", label: "Algorithms" },
                { value: "3", label: "Datasets" },
                { value: "25", label: "Runs" }
            ]
        }
    },
    hyperparameters: {
        intro: {
            terms: [
                { term: "Hyperparameter", def: "Settings configured before training, not learned from data" },
                { term: "Grid Search", def: "Exhaustive search over parameter combinations" },
                { term: "Learning Rate", def: "Step size for gradient descent optimization" },
                { term: "Regularization", def: "Penalty term to prevent overfitting" },
                { term: "Latent Factors", def: "Hidden dimensions in matrix factorization" }
            ],
            formulas: [
                { label: "MF Latent Factors", formula: "k ∈ {10, 20, 50, 100, 200}" },
                { label: "Learning Rate Range", formula: "α ∈ {0.001, 0.005, 0.01, 0.02}" },
                { label: "Regularization", formula: "λ ∈ {0.001, 0.01, 0.1}" }
            ],
            flow: ["Define Search Space", "Grid Search", "Train Models", "Evaluate", "Select Best"],
            examples: [
                { icon: "🎛️", text: "Too few factors → underfitting (misses patterns)" },
                { icon: "⚡", text: "High learning rate → unstable training" },
                { icon: "🛡️", text: "Strong regularization → prevents overfitting" }
            ],
            stats: [
                { value: "50", label: "Optimal Factors" },
                { value: "0.005", label: "Best Learn Rate" },
                { value: "0.02", label: "Best λ (reg)" },
                { value: "100", label: "Max Iterations" }
            ]
        },
        what_are_hyperparameters: {
            terms: [
                { term: "Parameter", def: "Learned from data during training (weights, biases)" },
                { term: "Hyperparameter", def: "Set before training, controls learning process" },
                { term: "Overfitting", def: "Model memorizes training data, fails on new data" },
                { term: "Underfitting", def: "Model too simple to capture patterns" }
            ],
            formulas: [
                { label: "Total Combinations", formula: "5 factors × 4 LR × 3 λ = 60 configs" },
                { label: "Time Complexity", formula: "O(k × n × m) per iteration" }
            ],
            flow: ["Choose Hyperparameters", "Train Model", "Validate", "Tune", "Final Model"],
            examples: [
                { icon: "🎚️", text: "Learning rate = gas pedal, regularization = brakes" },
                { icon: "🏠", text: "Like adjusting oven temp & time before baking" },
                { icon: "🎸", text: "Tuning a guitar before playing" }
            ],
            stats: [
                { value: "5", label: "Key Hyperparams" },
                { value: "60", label: "Combinations" },
                { value: "3x", label: "Performance Gain" },
                { value: "10-200", label: "Factor Range" }
            ]
        },
        sensitivity: {
            terms: [
                { term: "Sensitivity Analysis", def: "Study of how parameter changes affect output" },
                { term: "Optimal Range", def: "Parameter values giving best performance" },
                { term: "Plateau", def: "Region where changes have little effect" },
                { term: "Cliff", def: "Region where small changes cause big effects" }
            ],
            formulas: [
                { label: "Factors Sensitivity", formula: "10→50: -0.08 RMSE | 50→200: ±0.01" },
                { label: "Learning Rate", formula: "Sweet spot: 0.005-0.01" },
                { label: "Regularization", formula: "Too low: overfit | Too high: underfit" }
            ],
            flow: ["Vary One Parameter", "Fix Others", "Measure Output", "Plot Curve", "Find Optimum"],
            examples: [
                { icon: "📈", text: "Factors: Diminishing returns after 50" },
                { icon: "⚡", text: "Learning rate: Narrow optimal window" },
                { icon: "🎯", text: "K neighbors: Stable between 30-50" }
            ],
            stats: [
                { value: "50", label: "Optimal Factors" },
                { value: "0.005", label: "Best LR" },
                { value: "40", label: "Best K" },
                { value: "0.02", label: "Best λ" }
            ]
        },
        playground: {
            terms: [
                { term: "Interactive Tuning", def: "Real-time parameter adjustment and feedback" },
                { term: "Live Preview", def: "See effects of changes immediately" },
                { term: "Reset", def: "Return to default/optimal settings" },
                { term: "Export", def: "Save configuration for later use" }
            ],
            formulas: [
                { label: "Default Config", formula: "k=50, α=0.005, λ=0.02" },
                { label: "Speed vs Quality", formula: "Lower k = faster, higher k = better" }
            ],
            flow: ["Adjust Slider", "See Preview", "Compare Results", "Fine-tune", "Apply"],
            examples: [
                { icon: "🎮", text: "Try k=10 vs k=200 and see RMSE change" },
                { icon: "⚡", text: "Watch training converge in real-time" },
                { icon: "💾", text: "Save your best configuration" }
            ],
            stats: [
                { value: "5", label: "Tunable Params" },
                { value: "Real-time", label: "Feedback" },
                { value: "∞", label: "Experiments" },
                { value: "1-click", label: "Reset" }
            ]
        },
        key_insights: {
            terms: [
                { term: "Sweet Spot", def: "Optimal parameter value balancing tradeoffs" },
                { term: "Diminishing Returns", def: "Each increase helps less than the previous" },
                { term: "Default Values", def: "Good starting points for most cases" },
                { term: "Domain-Specific", def: "Optimal values may vary by dataset/domain" }
            ],
            formulas: [
                { label: "Key Finding 1", formula: "k=50 optimal across datasets" },
                { label: "Key Finding 2", formula: "LR 0.005-0.01 is safe range" },
                { label: "Key Finding 3", formula: "Regularization prevents 15% overfit" }
            ],
            flow: ["Start with Defaults", "Tune Most Sensitive", "Validate", "Fine-tune Others", "Document"],
            examples: [
                { icon: "💡", text: "Always start with k=50, then adjust" },
                { icon: "⚠️", text: "Never set learning rate > 0.1" },
                { icon: "✅", text: "Use regularization for sparse data" }
            ],
            stats: [
                { value: "3x", label: "Tuning Benefit" },
                { value: "50", label: "Default k" },
                { value: "0.005", label: "Safe LR" },
                { value: "15%", label: "Overfit Prevention" }
            ]
        }
    },
    algorithms: {
        collaborative_filtering: {
            terms: [
                { term: "User-User CF", def: "Finds users with similar rating patterns" },
                { term: "Item-Item CF", def: "Finds items rated similarly by users" },
                { term: "Cosine Similarity", def: "Measures angle between user/item vectors" },
                { term: "K-Nearest Neighbors", def: "Uses K most similar users/items" },
                { term: "Cold Start", def: "Problem with new users/items having no history" }
            ],
            formulas: [
                { label: "Cosine Similarity", formula: "cos(θ) = (A·B) / (||A|| ||B||)" },
                { label: "Prediction", formula: "r̂ᵤᵢ = r̄ᵤ + Σsim(u,v)(rᵥᵢ - r̄ᵥ) / Σ|sim|" },
                { label: "Performance", formula: "RMSE: 0.9234, Time: 2.3s" }
            ],
            flow: ["Build User-Item Matrix", "Compute Similarities", "Find Neighbors", "Aggregate Ratings", "Predict"],
            examples: [
                { icon: "👥", text: "Users who liked Matrix also liked Inception" },
                { icon: "🎬", text: "Find your 'rating twins' among millions of users" },
                { icon: "⚡", text: "Fast training but memory-intensive" }
            ],
            stats: [
                { value: "0.9234", label: "RMSE" },
                { value: "2.3s", label: "Train Time" },
                { value: "40", label: "Neighbors (K)" },
                { value: "93.7%", label: "Data Sparsity" }
            ]
        },
        matrix_factorization: {
            terms: [
                { term: "Latent Factors", def: "Hidden dimensions explaining user-item interactions" },
                { term: "SVD", def: "Singular Value Decomposition - matrix factorization technique" },
                { term: "Embedding", def: "Dense vector representation of users/items" },
                { term: "Gradient Descent", def: "Optimization algorithm to minimize error" }
            ],
            formulas: [
                { label: "Matrix Factorization", formula: "R ≈ P × Qᵀ (users × items)" },
                { label: "Prediction", formula: "r̂ᵤᵢ = pᵤ · qᵢ + bᵤ + bᵢ + μ" },
                { label: "Loss Function", formula: "L = Σ(rᵤᵢ - r̂ᵤᵢ)² + λ(||P||² + ||Q||²)" }
            ],
            flow: ["Initialize P, Q", "Predict Ratings", "Compute Error", "Update via SGD", "Converge"],
            examples: [
                { icon: "🏆", text: "Won the $1M Netflix Prize competition" },
                { icon: "🧠", text: "Discovers hidden 'taste dimensions'" },
                { icon: "📐", text: "User vector × Item vector = Rating" }
            ],
            stats: [
                { value: "0.8956", label: "Best RMSE" },
                { value: "50", label: "Latent Factors" },
                { value: "100", label: "Iterations" },
                { value: "0.02", label: "Regularization" }
            ]
        },
        neural_cf: {
            terms: [
                { term: "Neural Network", def: "Layered model that learns complex patterns" },
                { term: "Embedding Layer", def: "Converts IDs to dense vectors" },
                { term: "MLP", def: "Multi-Layer Perceptron - fully connected layers" },
                { term: "Non-linearity", def: "Activation functions enabling complex patterns" },
                { term: "Dropout", def: "Regularization by randomly disabling neurons" }
            ],
            formulas: [
                { label: "NCF Architecture", formula: "Embed → Concat → MLP → Output" },
                { label: "Hidden Layers", formula: "[64, 32, 16, 8]" },
                { label: "Activation", formula: "ReLU(x) = max(0, x)" }
            ],
            flow: ["User/Item IDs", "Embed Lookup", "Concatenate", "MLP Layers", "Rating/Rank"],
            examples: [
                { icon: "🧠", text: "Learns non-linear user-item interactions" },
                { icon: "🎯", text: "Best NDCG - excels at ranking" },
                { icon: "⚙️", text: "More complex but captures subtle patterns" }
            ],
            stats: [
                { value: "0.8534", label: "Best NDCG@10" },
                { value: "64", label: "Embedding Dim" },
                { value: "4", label: "Hidden Layers" },
                { value: "0.2", label: "Dropout Rate" }
            ]
        },
        content_based: {
            terms: [
                { term: "Item Features", def: "Attributes describing items (genre, author, etc.)" },
                { term: "TF-IDF", def: "Term Frequency-Inverse Document Frequency weighting" },
                { term: "Feature Vector", def: "Numerical representation of item attributes" },
                { term: "User Profile", def: "Aggregated preferences from liked items" }
            ],
            formulas: [
                { label: "TF-IDF", formula: "tf(t,d) × log(N/df(t))" },
                { label: "Similarity", formula: "cos(user_profile, item_features)" },
                { label: "Coverage", formula: "92% (Best in study)" }
            ],
            flow: ["Extract Features", "Build Item Profiles", "Aggregate User Likes", "Compute Similarity", "Recommend"],
            examples: [
                { icon: "📚", text: "Liked sci-fi? Here are more sci-fi books" },
                { icon: "🆕", text: "No cold start for new items - uses features immediately" },
                { icon: "🔍", text: "Explainable: 'Because you liked action movies'" }
            ],
            stats: [
                { value: "92%", label: "Coverage" },
                { value: "0.9856", label: "RMSE" },
                { value: "0", label: "Item Cold Start" },
                { value: "High", label: "Explainability" }
            ]
        },
        graph_based: {
            terms: [
                { term: "Bipartite Graph", def: "Network with users on one side, items on other" },
                { term: "Random Walk", def: "Simulated path through the graph" },
                { term: "PageRank", def: "Algorithm ranking nodes by importance" },
                { term: "Propagation", def: "Spreading information through network edges" }
            ],
            formulas: [
                { label: "PageRank", formula: "PR(u) = (1-d) + d × Σ PR(v)/L(v)" },
                { label: "Random Walk", formula: "P(next) ∝ edge_weight" },
                { label: "Damping Factor", formula: "d = 0.85 (standard)" }
            ],
            flow: ["Build Graph", "Initialize Scores", "Random Walks", "Propagate", "Rank Nodes"],
            examples: [
                { icon: "🕸️", text: "User → Movie → Similar User → Different Movie" },
                { icon: "🚶", text: "Where does a random walker end up most often?" },
                { icon: "🔗", text: "Captures indirect relationships" }
            ],
            stats: [
                { value: "0.9456", label: "RMSE" },
                { value: "0.85", label: "Damping" },
                { value: "100", label: "Walk Steps" },
                { value: "High", label: "Interpretability" }
            ]
        },
        hybrid: {
            terms: [
                { term: "Weighted Hybrid", def: "Combines predictions with learned weights" },
                { term: "Switching Hybrid", def: "Selects best method based on context" },
                { term: "Cascade Hybrid", def: "Refines one method's output with another" },
                { term: "Feature Augmentation", def: "Uses one method's output as features for another" }
            ],
            formulas: [
                { label: "Weighted Combination", formula: "r̂ = α·r̂_CF + (1-α)·r̂_CB" },
                { label: "Optimal Weight", formula: "α = 0.7 (CF weight)" },
                { label: "Hybrid RMSE", formula: "0.9089 (beats most single methods)" }
            ],
            flow: ["CF Prediction", "CB Prediction", "Combine Weights", "Meta-Learning", "Final Output"],
            examples: [
                { icon: "🎯", text: "Netflix uses 100+ models in their hybrid" },
                { icon: "💪", text: "CF strengths + CB strengths = Better overall" },
                { icon: "🛡️", text: "Robust to individual method failures" }
            ],
            stats: [
                { value: "0.9089", label: "RMSE" },
                { value: "0.7", label: "CF Weight" },
                { value: "0.3", label: "CB Weight" },
                { value: "2", label: "Methods Combined" }
            ]
        },
        svdpp: {
            terms: [
                { term: "Implicit Feedback", def: "User behavior signals (views, clicks, time spent)" },
                { term: "Explicit Feedback", def: "Direct ratings given by users" },
                { term: "SVD++", def: "SVD enhanced with implicit feedback signals" },
                { term: "Auxiliary Features", def: "Additional signals beyond ratings" }
            ],
            formulas: [
                { label: "SVD++ Prediction", formula: "r̂ᵤᵢ = μ + bᵤ + bᵢ + qᵢᵀ(pᵤ + |N(u)|^(-½)Σyⱼ)" },
                { label: "Implicit Term", formula: "|N(u)|^(-½) Σyⱼ for j∈N(u)" },
                { label: "Performance Gain", formula: "+2-3% over basic SVD" }
            ],
            flow: ["Collect Implicit", "Combine with Ratings", "Extended SVD", "Train", "Predict"],
            examples: [
                { icon: "👁️", text: "You viewed but didn't rate → still useful signal" },
                { icon: "⏱️", text: "Time spent on page indicates interest" },
                { icon: "📈", text: "More data = better predictions" }
            ],
            stats: [
                { value: "0.9012", label: "RMSE" },
                { value: "+3%", label: "vs Basic SVD" },
                { value: "2x", label: "Data Used" },
                { value: "50", label: "Factors" }
            ]
        },
        association_rules: {
            terms: [
                { term: "Support", def: "Frequency of itemset in all transactions" },
                { term: "Confidence", def: "P(B|A) - probability of B given A" },
                { term: "Lift", def: "How much more likely B is when A present" },
                { term: "Apriori", def: "Classic algorithm for finding frequent itemsets" }
            ],
            formulas: [
                { label: "Support", formula: "sup(A→B) = count(A∪B) / N" },
                { label: "Confidence", formula: "conf(A→B) = sup(A∪B) / sup(A)" },
                { label: "Lift", formula: "lift(A→B) = conf(A→B) / sup(B)" }
            ],
            flow: ["Find Frequent Items", "Generate Candidates", "Prune", "Extract Rules", "Filter by Metrics"],
            examples: [
                { icon: "🛒", text: "Beer + Diapers: Famous unexpected association" },
                { icon: "📦", text: "Bought laptop → likely to buy bag" },
                { icon: "🔍", text: "Discovers non-obvious patterns" }
            ],
            stats: [
                { value: "0.01", label: "Min Support" },
                { value: "0.5", label: "Min Confidence" },
                { value: "1.5", label: "Min Lift" },
                { value: "1000s", label: "Rules Found" }
            ]
        },
        popularity: {
            terms: [
                { term: "Popularity Baseline", def: "Recommend most popular items to everyone" },
                { term: "Non-Personalized", def: "Same recommendations for all users" },
                { term: "Benchmark", def: "Simple method to beat for justification" },
                { term: "Long Tail", def: "Less popular items that popularity misses" }
            ],
            formulas: [
                { label: "Popularity Score", formula: "pop(i) = count(ratings_i) / N" },
                { label: "Ranking", formula: "Sort items by popularity descending" },
                { label: "Baseline RMSE", formula: "~1.1 (global mean prediction)" }
            ],
            flow: ["Count Interactions", "Rank by Popularity", "Return Top-K", "Same for All Users"],
            examples: [
                { icon: "📈", text: "Everyone sees 'Most Popular Movies'" },
                { icon: "🎯", text: "Surprisingly effective for new users" },
                { icon: "📉", text: "Ignores personal taste entirely" }
            ],
            stats: [
                { value: "1.05", label: "RMSE" },
                { value: "0.65", label: "NDCG@10" },
                { value: "0", label: "Personalization" },
                { value: "5%", label: "Coverage" }
            ]
        },
        context_aware: {
            terms: [
                { term: "Context", def: "Circumstances affecting user preferences (time, location, mood)" },
                { term: "Temporal Dynamics", def: "How preferences change over time" },
                { term: "Session-Based", def: "Recommendations within a single session" },
                { term: "Multi-Armed Bandit", def: "Balancing exploration vs exploitation" }
            ],
            formulas: [
                { label: "Context Integration", formula: "r̂ᵤᵢₜ = f(user, item, context)" },
                { label: "Time Decay", formula: "weight = e^(-λ × age)" },
                { label: "Session Model", formula: "P(next|session_history)" }
            ],
            flow: ["Capture Context", "Encode Features", "Context-Aware Model", "Predict", "Adapt"],
            examples: [
                { icon: "🌅", text: "Morning: News & Coffee recommendations" },
                { icon: "🌙", text: "Evening: Entertainment & Relaxation" },
                { icon: "📍", text: "At gym: Workout music playlist" }
            ],
            stats: [
                { value: "0.9234", label: "RMSE" },
                { value: "+8%", label: "CTR Lift" },
                { value: "5", label: "Context Features" },
                { value: "High", label: "Relevance" }
            ]
        }
    }
};

// ============= VISUAL COMPANION FUNCTIONS =============

let visualCompanionEnabled = true;
let currentVisualization = null;
let vcProgressInterval = null;

// Draggable state
let vcDragState = {
    isDragging: false,
    startX: 0,
    startY: 0,
    startLeft: 0,
    startTop: 0
};

// Initialize draggable visual companion
function initDraggableVC() {
    const vc = document.getElementById('visual-companion');
    const header = document.getElementById('vc-header');
    if (!vc || !header) return;
    
    header.addEventListener('mousedown', vcDragStart);
    document.addEventListener('mousemove', vcDragMove);
    document.addEventListener('mouseup', vcDragEnd);
    
    // Touch support for mobile
    header.addEventListener('touchstart', vcTouchStart, { passive: false });
    document.addEventListener('touchmove', vcTouchMove, { passive: false });
    document.addEventListener('touchend', vcDragEnd);
}

function vcDragStart(e) {
    // Don't drag if clicking buttons
    if (e.target.closest('.vc-btn')) return;
    
    const vc = document.getElementById('visual-companion');
    if (!vc) return;
    
    vcDragState.isDragging = true;
    vcDragState.startX = e.clientX;
    vcDragState.startY = e.clientY;
    
    const rect = vc.getBoundingClientRect();
    vcDragState.startLeft = rect.left;
    vcDragState.startTop = rect.top;
    
    vc.classList.add('dragging');
    vc.style.transition = 'none';
}

function vcTouchStart(e) {
    if (e.target.closest('.vc-btn')) return;
    
    const touch = e.touches[0];
    const vc = document.getElementById('visual-companion');
    if (!vc) return;
    
    vcDragState.isDragging = true;
    vcDragState.startX = touch.clientX;
    vcDragState.startY = touch.clientY;
    
    const rect = vc.getBoundingClientRect();
    vcDragState.startLeft = rect.left;
    vcDragState.startTop = rect.top;
    
    vc.classList.add('dragging');
    vc.style.transition = 'none';
    e.preventDefault();
}

function vcDragMove(e) {
    if (!vcDragState.isDragging) return;
    
    const vc = document.getElementById('visual-companion');
    if (!vc) return;
    
    const deltaX = e.clientX - vcDragState.startX;
    const deltaY = e.clientY - vcDragState.startY;
    
    let newLeft = vcDragState.startLeft + deltaX;
    let newTop = vcDragState.startTop + deltaY;
    
    // Keep within viewport bounds
    const maxLeft = window.innerWidth - vc.offsetWidth;
    const maxTop = window.innerHeight - vc.offsetHeight;
    
    newLeft = Math.max(0, Math.min(newLeft, maxLeft));
    newTop = Math.max(0, Math.min(newTop, maxTop));
    
    vc.style.left = newLeft + 'px';
    vc.style.top = newTop + 'px';
    vc.style.right = 'auto';
}

function vcTouchMove(e) {
    if (!vcDragState.isDragging) return;
    
    const touch = e.touches[0];
    const vc = document.getElementById('visual-companion');
    if (!vc) return;
    
    const deltaX = touch.clientX - vcDragState.startX;
    const deltaY = touch.clientY - vcDragState.startY;
    
    let newLeft = vcDragState.startLeft + deltaX;
    let newTop = vcDragState.startTop + deltaY;
    
    // Keep within viewport bounds
    const maxLeft = window.innerWidth - vc.offsetWidth;
    const maxTop = window.innerHeight - vc.offsetHeight;
    
    newLeft = Math.max(0, Math.min(newLeft, maxLeft));
    newTop = Math.max(0, Math.min(newTop, maxTop));
    
    vc.style.left = newLeft + 'px';
    vc.style.top = newTop + 'px';
    vc.style.right = 'auto';
    
    e.preventDefault();
}

function vcDragEnd() {
    if (!vcDragState.isDragging) return;
    
    vcDragState.isDragging = false;
    
    const vc = document.getElementById('visual-companion');
    if (vc) {
        vc.classList.remove('dragging');
        vc.style.transition = '';
    }
}

function showVisualCompanion(page, scriptKey) {
    const vc = document.getElementById('visual-companion');
    if (!vc || !visualCompanionEnabled) return;
    
    // Get visualization data
    let vizData = null;
    if (visualCompanionData[page] && visualCompanionData[page][scriptKey]) {
        vizData = visualCompanionData[page][scriptKey];
    } else if (visualCompanionData.algorithms && visualCompanionData.algorithms[scriptKey]) {
        vizData = visualCompanionData.algorithms[scriptKey];
    }
    
    if (!vizData) {
        // Try to find any matching data
        for (const p in visualCompanionData) {
            if (visualCompanionData[p][scriptKey]) {
                vizData = visualCompanionData[p][scriptKey];
                break;
            }
        }
    }
    
    if (!vizData) {
        console.log('No visualization data for:', page, scriptKey);
        vc.classList.remove('active');
        return;
    }
    
    currentVisualization = vizData;
    
    // Update topic name
    const topicEl = document.getElementById('vc-topic-name');
    if (topicEl) {
        topicEl.textContent = currentTitle || scriptKey.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
    
    // Update terms
    const termsEl = document.getElementById('vc-terms');
    if (termsEl && vizData.terms) {
        termsEl.innerHTML = vizData.terms.map(t => 
            `<span class="vc-term">${t.term}<span class="term-def">${t.def}</span></span>`
        ).join('');
        termsEl.closest('.vc-section').classList.remove('hidden');
    } else if (termsEl) {
        termsEl.closest('.vc-section').classList.add('hidden');
    }
    
    // Update formulas
    const formulasEl = document.getElementById('vc-formulas');
    if (formulasEl && vizData.formulas) {
        formulasEl.innerHTML = vizData.formulas.map(f => 
            `<div class="vc-formula"><div class="vc-formula-label">${f.label}</div>${f.formula}</div>`
        ).join('');
        formulasEl.closest('.vc-section').classList.remove('hidden');
    } else if (formulasEl) {
        formulasEl.closest('.vc-section').classList.add('hidden');
    }
    
    // Update flowchart
    const flowEl = document.getElementById('vc-flowchart');
    if (flowEl && vizData.flow) {
        flowEl.innerHTML = vizData.flow.map((step, i) => 
            (i > 0 ? '<span class="vc-flow-arrow">↓</span>' : '') + 
            `<div class="vc-flow-node">${step}</div>`
        ).join('');
        flowEl.closest('.vc-section').classList.remove('hidden');
    } else if (flowEl) {
        flowEl.closest('.vc-section').classList.add('hidden');
    }
    
    // Update examples
    const examplesEl = document.getElementById('vc-examples');
    if (examplesEl && vizData.examples) {
        examplesEl.innerHTML = vizData.examples.map(e => 
            `<div class="vc-example"><span class="vc-example-icon">${e.icon}</span><span class="vc-example-text">${e.text}</span></div>`
        ).join('');
        examplesEl.closest('.vc-section').classList.remove('hidden');
    } else if (examplesEl) {
        examplesEl.closest('.vc-section').classList.add('hidden');
    }
    
    // Update stats
    const statsEl = document.getElementById('vc-stats');
    if (statsEl && vizData.stats) {
        statsEl.innerHTML = vizData.stats.map(s => 
            `<div class="vc-stat"><div class="vc-stat-value">${s.value}</div><div class="vc-stat-label">${s.label}</div></div>`
        ).join('');
        statsEl.closest('.vc-section').classList.remove('hidden');
    } else if (statsEl) {
        statsEl.closest('.vc-section').classList.add('hidden');
    }
    
    // Reset progress bar
    const progressBar = document.getElementById('vc-progress-bar');
    if (progressBar) progressBar.style.width = '0%';
    
    // Show panel
    vc.classList.add('active');
    vc.classList.remove('minimized');
    
    // Start progress animation
    startVCProgress();
}

function startVCProgress() {
    if (vcProgressInterval) clearInterval(vcProgressInterval);
    
    const progressBar = document.getElementById('vc-progress-bar');
    if (!progressBar) return;
    
    let progress = 0;
    const totalChunks = speechQueue.length + 1; // +1 for current chunk
    
    vcProgressInterval = setInterval(() => {
        if (!isPlaying) {
            clearInterval(vcProgressInterval);
            return;
        }
        
        const currentChunk = totalChunks - speechQueue.length;
        progress = (currentChunk / totalChunks) * 100;
        progressBar.style.width = Math.min(progress, 100) + '%';
        
        if (progress >= 100) {
            clearInterval(vcProgressInterval);
        }
    }, 500);
}

function hideVisualCompanion() {
    const vc = document.getElementById('visual-companion');
    if (vc) vc.classList.remove('active');
    if (vcProgressInterval) clearInterval(vcProgressInterval);
}

function toggleVisualCompanion() {
    const vc = document.getElementById('visual-companion');
    if (vc) vc.classList.toggle('minimized');
}

function closeVisualCompanion() {
    hideVisualCompanion();
    visualCompanionEnabled = false;
    // Re-enable after 30 seconds
    setTimeout(() => { visualCompanionEnabled = true; }, 30000);
}

// ============= SPEECH SYNTHESIS ENGINE =============
// Enhanced with better voice initialization, error handling, and text chunking

let synth = null;
let voices = [];
let currentVoice = null;
let currentUtterance = null;
let isPlaying = false;
let isPaused = false;
let isMuted = false;
let speechRate = 1.0;
let captionsEnabled = true;
let currentPage = 'overview';
let voicesLoaded = false;
let speechQueue = []; // Queue for chunked text
let currentTitle = ''; // Store current narration title
let speechSessionId = 0; // Unique ID for each narration session to prevent race conditions
let currentScriptKey = ''; // Track current script for visualization

// Chrome has a bug where utterances > ~200 words fail silently
// This function chunks text at sentence boundaries
function chunkText(text, maxWords = 150) {
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];
    let currentChunk = '';
    let currentWordCount = 0;
    
    for (const sentence of sentences) {
        const sentenceWords = sentence.trim().split(/\s+/).length;
        
        if (currentWordCount + sentenceWords > maxWords && currentChunk) {
            chunks.push(currentChunk.trim());
            currentChunk = sentence;
            currentWordCount = sentenceWords;
        } else {
            currentChunk += ' ' + sentence;
            currentWordCount += sentenceWords;
        }
    }
    
    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }
    
    return chunks.length > 0 ? chunks : [text];
}

// Initialize speech synthesis
function initNarration() {
    console.log('Initializing narration system...');
    
    // Check if speech synthesis is available
    if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported in this browser');
        const status = document.getElementById('narration-status');
        if (status) {
            status.textContent = 'Not Supported';
            status.className = 'narration-status paused';
        }
        return;
    }
    
    synth = window.speechSynthesis;
    
    // Load voices with retry mechanism
    loadVoicesWithRetry();
    
    // Chrome requires this event listener
    if (synth.onvoiceschanged !== undefined) {
        synth.onvoiceschanged = function() {
            console.log('Voices changed event fired');
            loadVoices();
        };
    }
    
    // Add page change listeners
    setupPageListeners();
    
    // Initialize draggable visual companion
    initDraggableVC();
    
    console.log('Narration system initialized');
}

function loadVoicesWithRetry(retryCount = 0) {
    const maxRetries = 10;
    const retryDelay = 100;
    
    const availableVoices = synth.getVoices();
    
    if (availableVoices.length > 0) {
        loadVoices();
    } else if (retryCount < maxRetries) {
        console.log(`Waiting for voices... attempt ${retryCount + 1}`);
        setTimeout(() => loadVoicesWithRetry(retryCount + 1), retryDelay);
    } else {
        console.warn('Could not load voices after retries');
    }
}

function loadVoices() {
    if (!synth) return;
    
    const allVoices = synth.getVoices();
    console.log(`Found ${allVoices.length} total voices`);
    
    // Filter to English voices
    voices = allVoices.filter(v => v.lang.startsWith('en'));
    console.log(`Found ${voices.length} English voices`);
    
    if (voices.length === 0) {
        voices = allVoices; // Use all voices if no English found
    }
    
    // Select preferred voice
    const preferredVoices = [
        'Google UK English Female',
        'Google US English',
        'Microsoft Zira',
        'Microsoft David',
        'Samantha',
        'Alex',
        'Daniel',
        'Karen',
        'Moira'
    ];
    
    currentVoice = null;
    for (const name of preferredVoices) {
        const voice = voices.find(v => v.name.includes(name));
        if (voice) {
            currentVoice = voice;
            console.log(`Selected preferred voice: ${voice.name}`);
            break;
        }
    }
    
    if (!currentVoice && voices.length > 0) {
        currentVoice = voices[0];
        console.log(`Selected default voice: ${currentVoice.name}`);
    }
    
    // Populate voice selector
    const selector = document.getElementById('voice-selector');
    if (selector && voices.length > 0) {
        selector.innerHTML = voices.map((v, i) => 
            `<option value="${i}" ${v === currentVoice ? 'selected' : ''}>${v.name} (${v.lang})</option>`
        ).join('');
    }
    
    voicesLoaded = true;
    updateNarrationUI();
}

function setupPageListeners() {
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const onclickAttr = this.getAttribute('onclick');
            if (onclickAttr) {
                const pageMatch = onclickAttr.match(/showPage\(['"](.+)['"]\)/);
                if (pageMatch) {
                    currentPage = pageMatch[1];
                    console.log(`Page changed to: ${currentPage}`);
                    
                    // Hide scene nav when changing pages
                    const sceneNav = document.getElementById('scene-nav');
                    if (sceneNav) sceneNav.style.display = 'none';
                    
                    setTimeout(() => {
                        const autoNarrate = document.getElementById('auto-narrate-toggle');
                        if (autoNarrate && autoNarrate.checked) {
                            narrateCurrentPage();
                        }
                    }, 500);
                }
            }
        });
    });
}

function setNarrationVoice(index) {
    if (voices[index]) {
        currentVoice = voices[index];
        console.log(`Voice changed to: ${currentVoice.name}`);
    }
}

function updateNarrationSpeed(value) {
    speechRate = parseFloat(value);
    const display = document.getElementById('speed-value');
    if (display) display.textContent = value + 'x';
}

function speak(text, title = 'Narration', page = null, scriptKey = null) {
    if (!synth) {
        console.error('Speech synthesis not initialized');
        alert('Speech synthesis is not available. Please try a different browser like Chrome.');
        return;
    }
    
    if (isMuted) {
        console.log('Muted - showing captions only');
        if (captionsEnabled) showCaptions(title, text);
        return;
    }
    
    if (!voicesLoaded || voices.length === 0) {
        console.log('Voices not loaded yet, retrying...');
        loadVoicesWithRetry();
        setTimeout(() => speak(text, title, page, scriptKey), 500);
        return;
    }
    
    // Stop any current speech and increment session ID
    synth.cancel();
    speechQueue = [];
    speechSessionId++; // New session - invalidates any pending callbacks from old session
    const thisSessionId = speechSessionId;
    
    console.log(`Starting new speech session ${thisSessionId}: "${title}"`);
    
    isPlaying = false;
    isPaused = false;
    
    // Store title and script key for visualization
    currentTitle = title;
    currentScriptKey = scriptKey || '';
    
    if (captionsEnabled) {
        showCaptions(title, text);
    }
    
    // Show visual companion with relevant data
    if (page && scriptKey) {
        showVisualCompanion(page, scriptKey);
    } else if (scriptKey) {
        // Try to find visualization data by scriptKey alone
        showVisualCompanion(currentPage, scriptKey);
    }
    
    // Chunk the text to avoid Chrome's utterance length bug
    speechQueue = chunkText(text);
    console.log(`Speaking "${title}" in ${speechQueue.length} chunks (session ${thisSessionId})`);
    
    // Start speaking the queue
    speakNextChunk(thisSessionId);
}

function speakNextChunk(sessionId) {
    // Check if this session is still valid (user hasn't started a new narration)
    if (sessionId !== speechSessionId) {
        console.log(`Session ${sessionId} expired, current is ${speechSessionId}. Stopping.`);
        return;
    }
    
    if (speechQueue.length === 0) {
        console.log('Speech queue complete');
        isPlaying = false;
        isPaused = false;
        updateNarrationUI();
        return;
    }
    
    const chunk = speechQueue.shift();
    currentUtterance = new SpeechSynthesisUtterance(chunk);
    
    if (currentVoice) {
        currentUtterance.voice = currentVoice;
    }
    currentUtterance.rate = speechRate;
    currentUtterance.pitch = 1.0;
    currentUtterance.volume = 1.0;
    
    currentUtterance.onstart = () => {
        console.log(`Chunk started (session ${sessionId}), remaining:`, speechQueue.length);
        isPlaying = true;
        isPaused = false;
        updateNarrationUI();
    };
    
    currentUtterance.onend = () => {
        console.log(`Chunk ended (session ${sessionId})`);
        // Small delay between chunks for natural pacing
        // Pass session ID to verify this callback is still valid
        setTimeout(() => speakNextChunk(sessionId), 100);
    };
    
    currentUtterance.onerror = (event) => {
        console.error('Speech error:', event.error);
        // Only continue if not canceled and session is still valid
        if (event.error !== 'canceled' && sessionId === speechSessionId) {
            setTimeout(() => speakNextChunk(sessionId), 100);
        } else {
            isPlaying = false;
            isPaused = false;
            updateNarrationUI();
        }
    };
    
    currentUtterance.onpause = () => {
        isPaused = true;
        updateNarrationUI();
    };
    
    currentUtterance.onresume = () => {
        isPaused = false;
        updateNarrationUI();
    };
    
    // Speak - small delay for Chrome stability
    setTimeout(() => {
        // Double-check session is still valid before speaking
        if (sessionId === speechSessionId) {
            synth.speak(currentUtterance);
            console.log(`Speaking chunk (session ${sessionId}):`, chunk.substring(0, 50) + '...');
        }
    }, 50);
}

function narrationToggle() {
    if (!synth) return;
    
    if (isPlaying && !isPaused) {
        synth.pause();
        console.log('Paused');
    } else if (isPaused) {
        synth.resume();
        console.log('Resumed');
    } else {
        // Nothing playing, start current page narration
        narrateCurrentPage();
    }
}

function narrationStop() {
    if (synth) {
        synth.cancel();
    }
    speechQueue = []; // Clear the queue
    isPlaying = false;
    isPaused = false;
    hideCaptions();
    hideVisualCompanion(); // Hide visual companion when stopping
    updateNarrationUI();
}

function toggleNarrationMute() {
    isMuted = !isMuted;
    if (isMuted && isPlaying) narrationStop();
    const btn = document.getElementById('narration-mute-btn');
    if (btn) btn.innerHTML = isMuted ? '🔇' : '🔊';
}

function toggleNarrationCaptions() {
    const toggle = document.getElementById('captions-toggle');
    captionsEnabled = toggle ? toggle.checked : true;
    if (!captionsEnabled) hideCaptions();
}

function toggleNarrationPanel() {
    const panel = document.getElementById('narration-panel');
    if (!panel) return;
    
    panel.classList.toggle('minimized');
    
    if (panel.classList.contains('minimized')) {
        panel.onclick = function(e) {
            if (e.target === panel) {
                panel.classList.remove('minimized');
                panel.onclick = null;
            }
        };
    }
}

function updateNarrationUI() {
    const playBtn = document.getElementById('narration-play-btn');
    const status = document.getElementById('narration-status');
    
    if (playBtn) {
        playBtn.innerHTML = (isPlaying && !isPaused) ? '⏸️' : '▶️';
    }
    
    if (status) {
        if (!synth || !voicesLoaded) {
            status.textContent = 'Loading...';
            status.className = 'narration-status';
        } else if (isPlaying && !isPaused) {
            status.textContent = 'Speaking...';
            status.className = 'narration-status speaking';
        } else if (isPaused) {
            status.textContent = 'Paused';
            status.className = 'narration-status paused';
        } else {
            status.textContent = 'Ready';
            status.className = 'narration-status ready';
        }
    }
}

function showCaptions(title, text) {
    const overlay = document.getElementById('caption-overlay');
    if (!overlay) return;
    
    const titleEl = overlay.querySelector('.caption-title');
    const textEl = document.getElementById('caption-text');
    
    if (titleEl) titleEl.textContent = title;
    if (textEl) textEl.textContent = text;
    overlay.classList.add('active');
}

function hideCaptions() {
    const overlay = document.getElementById('caption-overlay');
    if (overlay) overlay.classList.remove('active');
}

function narrateCurrentPage() {
    const pageMap = {
        'overview-page': 'overview',
        'visualizations-page': 'visualizations',
        'comparison-page': 'comparison',
        'results-page': 'results',
        'hyperparameters-page': 'hyperparameters'
    };
    
    const pageName = pageMap[currentPage] || currentPage;
    const scripts = narrationScripts[pageName];
    
    console.log(`Narrating page: ${pageName}`);
    
    if (scripts) {
        // Try intro first, then welcome
        if (scripts.intro) {
            speak(scripts.intro.text, scripts.intro.title, pageName, 'intro');
        } else if (scripts.welcome) {
            speak(scripts.welcome.text, scripts.welcome.title, pageName, 'welcome');
        } else {
            // Get first available script
            const firstKey = Object.keys(scripts)[0];
            if (firstKey) {
                speak(scripts[firstKey].text, scripts[firstKey].title, pageName, firstKey);
            }
        }
    } else {
        console.warn(`No scripts found for page: ${pageName}`);
    }
}

function narrateScript(page, scriptKey) {
    const scripts = narrationScripts[page];
    if (scripts && scripts[scriptKey]) {
        speak(scripts[scriptKey].text, scripts[scriptKey].title, page, scriptKey);
    } else {
        console.warn(`Script not found: ${page}/${scriptKey}`);
    }
}

function narrateAlgorithm(algorithmKey) {
    const script = narrationScripts.algorithms[algorithmKey];
    if (script) {
        speak(script.text, script.title, 'algorithms', algorithmKey);
    } else {
        console.warn(`Algorithm script not found: ${algorithmKey}`);
    }
}

function showSceneSelector() {
    const sceneNav = document.getElementById('scene-nav');
    if (!sceneNav) return;
    
    const pageMap = {
        'overview-page': 'overview',
        'visualizations-page': 'visualizations',
        'comparison-page': 'comparison',
        'results-page': 'results',
        'hyperparameters-page': 'hyperparameters'
    };
    
    const pageName = pageMap[currentPage] || currentPage;
    const scripts = narrationScripts[pageName];
    
    if (!scripts || Object.keys(scripts).length === 0) {
        sceneNav.innerHTML = '<p style="color: var(--text-muted); padding: 10px;">No topics available for this page</p>';
        sceneNav.style.display = 'block';
        return;
    }
    
    sceneNav.innerHTML = Object.entries(scripts).map(([key, script]) => 
        `<button class="scene-btn" onclick="narrateScript('${pageName}', '${key}')">${script.title}</button>`
    ).join('');
    
    sceneNav.style.display = sceneNav.style.display === 'none' || sceneNav.style.display === '' ? 'flex' : 'none';
}

function showAlgorithmSelector() {
    const sceneNav = document.getElementById('scene-nav');
    if (!sceneNav) return;
    
    const algorithms = narrationScripts.algorithms;
    
    sceneNav.innerHTML = Object.entries(algorithms).map(([key, script]) => 
        `<button class="scene-btn" onclick="narrateAlgorithm('${key}')">${script.title.split(' - ')[0]}</button>`
    ).join('');
    
    sceneNav.style.display = sceneNav.style.display === 'none' || sceneNav.style.display === '' ? 'flex' : 'none';
}

// Initialize when DOM is ready
if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initNarration);
    } else {
        // DOM already loaded
        setTimeout(initNarration, 100);
    }
}

// Export for global access (safe assignment pattern)
(function() {
    if (typeof window === 'undefined') return;
    
    const exports = {
        narrationScripts: narrationScripts,
        visualCompanionData: visualCompanionData,
        narrateAlgorithm: narrateAlgorithm,
        narrateScript: narrateScript,
        narrateCurrentPage: narrateCurrentPage,
        speak: speak,
        speakNextChunk: speakNextChunk,
        chunkText: chunkText,
        narrationStop: narrationStop,
        narrationToggle: narrationToggle,
        toggleNarrationMute: toggleNarrationMute,
        toggleNarrationCaptions: toggleNarrationCaptions,
        toggleNarrationPanel: toggleNarrationPanel,
        setNarrationVoice: setNarrationVoice,
        updateNarrationSpeed: updateNarrationSpeed,
        showSceneSelector: showSceneSelector,
        showAlgorithmSelector: showAlgorithmSelector,
        initNarration: initNarration,
        initDraggableVC: initDraggableVC,
        showVisualCompanion: showVisualCompanion,
        hideVisualCompanion: hideVisualCompanion,
        toggleVisualCompanion: toggleVisualCompanion,
        closeVisualCompanion: closeVisualCompanion
    };
    
    Object.keys(exports).forEach(function(key) {
        try {
            window[key] = exports[key];
        } catch (e) {
            console.warn('Could not export ' + key + ':', e);
        }
    });
})();
