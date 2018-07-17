---
layout: post
title:  "Making a Dataset"
date:   2018-07-17
visible: true
---

TL;DR: We find a way to scrape a lot of images from Reddit, and then format them to a standard 128x128 size. Code <a href="https://github.com/yangalexandery/blog-ws/blob/master/implementing-GANs/scraper.py">here</a>.

Yesterday, inspired by <a href="https://www.reddit.com/r/MachineLearning/comments/8vbkti/p_progan_trained_on_rearthporn_images/">this post</a> I wanted to try my own hand at writing a Generative Adversarial Network (GAN) and generating synthetic images. I figured I could just re-use their code to obtain my own dataset of images to train on.

### Problem 1
<a href="https://github.com/perplexingpegasus/ProGAN/blob/master/scripts/downloader.py">The script</a> used to download images from Reddit no longer worked, due to a performance optimization made by Reddit's team. Originally, Reddit would allow a user to view arbitrarily many submissions by simply clicking 'Next'. However, now Reddit caches the first 1000 submissions for every category, and doesn't allow immediate access to any other submissions. In other words, regardless of whether a user sorts by 'New' or 'Hot' or 'Top', Reddit will always show at most 1000 posts.

Up to a year ago, developers were able to get around this by making <a href="https://www.reddit.com/r/reddittips/comments/2ix73n/use_cloudsearch_to_search_for_posts_on_reddit/">more complex search queries</a>. Users were still able to use Reddit's search function to obtain a list of posts which were submitted within a range of time. As a result, a scraper could theoretically make a search query for as many time intervals as wanted, and therefore access as many submissions as wanted.

### Problem 2
<a href="https://www.reddit.com/r/changelog/comments/694o34/reddit_search_performance_improvements/">Recent improvements</a> to Reddit's search function removed the feature which allowed search queries on timestamp, and as a result this method of obtaining posts no longer worked. Luckily, <a href="https://pushshift.io/">pushshift.io</a> has been collecting Reddit data for a while, and allows for timestamp-based queries. Using their API, we're able to access URLs for as many images as we want.

### Problem 3
After downloading a lot of images and resizing them, I found out that quite a few images uploaded to Reddit look like this:
<img src="../../../images/making_dataset_1.png" style="display: block; margin: 0 auto;">
To get around this, we can just check the image size of what we download and make sure that the dimensions don't match with this specific image's dimensions, which are 130x60.

### End Result
Now that all these problems are fixed, We're able to collect a large number of images from the subreddit of our choice. My scraper code can be found <a href="https://github.com/yangalexandery/blog-ws/blob/master/implementing-GANs/scraper.py">here</a>. We resize each of them to fit within a 128x128 square, and zero-pad the images when necessary. I chose to scrape the subreddit <a href="https://www.reddit.com/r/CatsStandingUp/">/r/CatsStandingUp</a> as an example, and now I have a collection of 10,000 images of standing cats. We'll see soon if a GAN is able to learn from these images and synthesize new ones.