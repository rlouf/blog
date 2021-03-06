---
title: "The birth of an ETL pipeline"
date: 2018-08-09T22:11:10+01:00
draft: true
---
￼
In a digital world, data pipelines carry the blood of our applications: information. And like their physical
counterpart, data pipelines get messy when they break.

When everything is in thr same database, then great, you don't have to write any code to move data from one
place to another. Worse case scenario you have to create materialized views, but the related SQL queries are
easy to maintain. This design choice may not be feasible/desirable for many reasons, but as far as data
pipelining goes, it's ideal. The best data pipeline is the pipeline that doesn't exist.

## Enters the analytics

Everything works fine in the backend. Postgres does wonder and you can store tens of millions of user
properties, truckloads of informations about tracks, artists, etc and the social network. In the meantime, you
had the great idea to record actions of the users and send them to a third party for analysis and everything
worked fine.

Comes your first person who is officially in charge of the data. That would be me. Very quickly, I found
myself uncomfortable with the third party tool. It is a great product, and works really well for simple stuff,
but their advanced functionalities were kind of limited (no, I don't like K-means), and I wanted to do some
advanced analysis (Machine Learning TM) and needed to cross information with the social network anyway. In
short, you need to be able to write some SQL.

First you inquire into the third party's (paying of course) solution to access raw data. It's too expensive.
You need something cheaper, and you need it quick. Luckily, you've hacked around their export API a few months
ago, both driven by the desire to learn Go, and late-night-caffeine-induced paranoia. So I figured I could
just hack something quickly that will do the job.

Let me stop here for a second.

Don't do it quickly, do it right. **The 'quick hack' is not a quick hack, it is a first stab at a
central-to-be service**. Dependencies are graphs, and graphs grow from existing leaves. Chances are, if you
are building the first branches, you are building the trunk. Don't mess this up.

The ideal scenario would have been to implement a data ingestion server to which clients would have sent the
data, and which would have sent it to the database. But this was too time-consuming, and we were a 12 people
team trying to get to product-market-fit as fast as we could. Plus we were going to keep our third-party tool
for a while, so we figured that would be easiest solution. This was not the best solution, but I got to learn
a lot in the process.

## So how do you do this?

Your analyses down the line will depend on the quality of the transfer, and you cannot afford being worried
about reliability all the time. It needs to run so well that you end up forgetting about it. And when it
crashes, or if the machine is restarted, you don't want to have to intervene---every human interaction is
error-prone. The design requirements are:

1. No manual intervention in case of interruption. The program should know where ot restart; 
2. You must be certain that no data is missing at any point;
3. The system needs to be redeployed quickly in case of failure;

I settled on Golang as a language for the service. Golang has many qualities, and two important ones in my
case: it was designe for http transactions, and Go programs compile to a single binary that you can just
`scp` in you remote server for deployment. Plus, whether you like it or not, the way it handles error makes
your programs very reliable and easy to debug.

To fulfill (1) and (2), the program needs to keep some sort of state. But what happens if the program is
killed? Yes, state is lost. Manual intervention required. Ok, but what if you keep the state on the disk? If
you accidentally delete the instance, state is lost. So we came up with a different approach: centralized
state accounting. 

The way this works is the following. We create a database with a table name `etl_state` with the following
scheme:

```
CREATE TABLE etl_state (
  archive STRING PRIMARY KEY,
  import_date TIMESTAMP,
  imported BOOLEAN DEFAULT FALSE
);
```

`archive` is the atomic unit of data we need to get. In case of a daily batch, `archive` will contain a string
of the form `20181101`. `imported` is self-explanatory: when your ETL process has successfully moved the data,
you switch this one to `true`.

`import_date` serves 2 purposes here:

1. Making sure that two concurrent processes won't try to import the same data. When one process starts the
   import, it sets the import date as the current time, and other processes won't process this item unless the
   date is X hours ago. Set X to 10x whatever amount of time imports are supposed to take;
2. If the process that started the import fails for some exterior reason, another one will pick it up later,
   after X hours.

This is a very simple, but powerful scheme. I've had process getting killed for random reasons, and I can't
exlain to you the please of seeing the machine going back to the last stable state and starting the transfer
again, without needing an intervention on my part. It works a bit like a ratcher: it moves from one stable
state to another, and knows where to pick up if you haven't reached the next tooth. 

I now use this pattern for all critical ETL-related tasks. It also works wonders for more complicated ETL
tasks with dependencies. Consider the following pipeline I used to compute timeslices of our graph. It works
that way:

1. Compute the relationships formed the past day;
2. Compute the new mutual relationships formed;
3. Find the active users that data;
4. Compute timeslices

The dependency graph is as follows:

```
1 ---> 2 ---+
            |---> 4
       3 ---+
```

You can create 4 new tables, spawn 4 different processes that continuously probe the last processes' state and
does the processing. Mind you, the success of process (3) depends on the import we described earlier. You can
write the dependency down too, even if they run on different machines.

Why would you have processes running on different machines? Well, the idea is that the result of (0) may be
used elsewhere for other services, so you don't really want crashes or deployment in either of the processes
on this machine to affect this one in particular. In the example above, (1,2,3) are merely incremental steps
towards (4) that are not being used anywhere else.

You can already see where this is going:

- Messy codebase as different processes live in their own repository; Different coding styles;
- Hard-to-follow dependencies. Someone may discontinue 0 without knowing that (4) will be affected. Not likely
  in our 12-people team; highly likely in a large organization;
- A lot of boilerplate code to re-write e.v.e.r.y.t.i.m.e.

The ideal system would be a centralized codebase that represents the ETL processes as a Directed Acyclic
Graph, a graph that is explicit, visible, and subject to code review. I'm still uncomfortable at the idea of a
stateful program, so the state would have to live in a database that is automatically created upon the
creation of a new node. Alo, I don't want every deployment Fearing the silent death of goroutines, I will
probably switch to a language more adapted to the task at hand: Erlang, or Elixir. Hot code reloading seem
adapted to the problem.

When the program restarts, I don't want to have to wait for (3) to finish its new batch for (4) to start
catching up. I also don't want to lose the time (3) took everytime (plus 3 might be involved somewhere else,
messy ramifications).

## Addendum: why a new library?

There are two great pyton libraries that are designed to do exactly that: Luigi and Apache Ariflow. Why
another one? First size:

- Airflow: 97,000 lines of python code
- Luigi: 40,000 lines of python code

They pack a lot of nice features, many of which I will never use. There are also features that I would
probably need but are not in the libraries yet. This means a lot of time spent reading the codebase, instead
of learning a new old language.

Plus, using the ratchets I can separate my graphs in different codebases, and deploy them in different
servers. Since the state is managed remotely, no need to worry about it! d
