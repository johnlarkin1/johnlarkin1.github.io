---
title: Scylla-Go-Round
layout: post
featured-img: scylla-go-round
categories: [Development, Go, ScyllaDB]
summary: Automatically convert your Cassandra/Scylla DB types to static Go types. Download on Pypi now!
---

Woof that does not look like the Golang Gopher or the ScyllaDB monster.

Regardless! Check out this project [here][pypi-sgr]! Feel free to start utilizing it in your workflows and create issues if types are not supported or you find bugs. You can also find it [here][piwheels-sgr] on piwheels.

![sgr-pypi](/images/scylla-go-round/scylla-pypi.png){: .center }

![sgr-repo](/images/scylla-go-round/scylla-go-round-repo.png){: .center }

This project will help you to automatically generate Golang types from your Cassandra or ScyllaDB schema.

- [Motivation](#motivation)
- [Prerequisites](#prerequisites)
- [Example Walk Through](#example-walk-through)
  - [Starting Scylla Locally](#starting-scylla-locally)
  - [Creating a ScyllaDB Schema](#creating-a-scylladb-schema)
  - [Auto-generating the Go Types](#auto-generating-the-go-types)
- [Conclusion](#conclusion)

# Motivation

Recently, I've been working a lot in [Go][go] and also utilizing our [ScyllaDB][scylla] NoSQL databases for low latency, performant writes/reads.

[Go][go] offers many benefits over Python including concurrency and parallelism being a first class feature of the language. It also offers static typing which I'm generally a fan of. That being said, for writing to [ScyllaDB][scylla] from Go, there are a couple of headaches. So I built a tool to autogenerate your Go types from your ScyllaDB / CQL schema.

# Prerequisites

Note, if you're about to start shipping production Go code that works with Scylla, I highly encourage you to read these articles first:

- [**Golang and ScyllaDB Part 1: Introduction**][goscylla-pt1]
- [**Golang and ScyllaDB Part 2: Data Types**][goscylla-pt2]
- [**Golang and ScyllaDB Part 3: GoCQLX**][goscylla-pt3]

I would highly recommend reading and using [GoCQLX][gocqlx]. They have a ton of niceties around query building and binding to ensure smooth and performant queries or inserts. I have found their support around batch / bulk operations slightly cumbersome, but I think that was given some recent changes to revert back to using `gocql`'s Batch templating. See this comment [here](https://pkg.go.dev/github.com/scylladb/gocqlx/v2/qb#BatchBuilder):

> Deprecated: Please use gocql.Session.NewBatch() instead.

That being said, the one thing they do not support is automatic type generation. Manually building these static Go types is both error-prone and slow, two things that I do not like.

# Example Walk Through

In this section, we're going to walk through:

**CHECKLIST**

- ☑️ starting a single ScyllaDB instance locally (I'm assuming you have Docker Desktop installed)
- ☑️ creating a new Scylla schema with multiple tables
- ☑️ auto-generating the Scylla types for go from this

## Starting Scylla Locally

The beginning part of this is going to be a bit similar to this series [**ScyllaDB University: Install and Start ScyllaDB**](https://university.scylladb.com/lab-install-and-start-scylladb-part-1-of-2/).

Here's a code walk through:

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round ‹main›
╰─➤  docker run --name scyllaU -d scylladb/scylla:5.2.0 --overprovisioned 1 --smp 1

Unable to find image 'scylladb/scylla:5.2.0' locally
5.2.0: Pulling from scylladb/scylla
cd741b12a7ea: Pull complete
e3f1f254bd77: Downloading [============>                                      ]  101.4MB/392MB

...

╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round ‹main›
╰─➤  docker exec -it scyllaU nodetool status

Datacenter: datacenter1
=======================
Status=Up/Down
|/ State=Normal/Leaving/Joining/Moving
--  Address     Load       Tokens       Owns    Host ID                               Rack
UN  172.17.0.2  204 KB     256          ?       ea3ed2c5-c595-48d2-bf89-4388b1f20bf2  rack1

Note: Non-system keyspaces don't have the same replication settings, effective ownership information is meaningless

What's next:
    Try Docker Debug for seamless, persistent debugging tools in any container or image → docker debug scyllaU
    Learn more at https://docs.docker.com/go/debug-cli/
```

Ok fantastic! So now:

**CHECKLIST**

- ✅ starting a single ScyllaDB instance locally (I'm assuming you have Docker Desktop installed)
- ☑️ creating a new Scylla schema with multiple tables
- ☑️ auto-generating the Scylla types for go from this

## Creating a ScyllaDB Schema

Let's move on to the second part:

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round ‹main›
╰─➤  docker exec -it scyllaU cqlsh

Connected to  at 172.17.0.2:9042.
[cqlsh 5.0.1 | Cassandra 3.0.8 | CQL spec 3.3.1 | Native protocol v4]
Use HELP for help.
cqlsh>

CREATE KEYSPACE dog_park
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Table for storing dog profiles
CREATE TABLE dog_park.dog_profiles (
    dog_id UUID PRIMARY KEY,
    name TEXT,
    breed TEXT,
    age INT,
    weight DECIMAL,
    attributes MAP<TEXT, TEXT>,       -- Complex attributes (e.g., color, coat type)
    vaccination_dates LIST<TIMESTAMP>, -- List of vaccination dates
    friends SET<UUID>                 -- Set of friend dog IDs
);

-- Table for storing owner information
CREATE TABLE dog_park.owner_profiles (
    owner_id UUID PRIMARY KEY,
    name TEXT,
    contact_info MAP<TEXT, TEXT>,   -- Contact info with key-value pairs (e.g., phone, email)
    dog_ids SET<UUID>               -- Set of owned dog IDs
);

-- Table for storing dog activities
CREATE TABLE dog_park.dog_activities (
    activity_id UUID PRIMARY KEY,
    dog_id UUID,
    activity_type TEXT,             -- Type of activity (e.g., walk, training, play)
    activity_details TEXT,
    activity_date TIMESTAMP,
    tags SET<TEXT>                   -- Tags related to the activity (e.g., 'fun', 'exercise')
);

-- Table for storing grooming sessions
CREATE TABLE dog_park.grooming_sessions (
    session_id UUID PRIMARY KEY,
    dog_id UUID,
    groomer_name TEXT,
    session_date TIMESTAMP,
    services MAP<TEXT, TEXT>,       -- Map of services provided (e.g., bath, nail trim)
    notes TEXT                      -- Additional notes
);

cqlsh> SELECT table_name
FROM system_schema.tables
WHERE keyspace_name = 'dog_park';

 table_name
-------------------
    dog_activities
      dog_profiles
 grooming_sessions
    owner_profiles
```

![table-list](/images/scylla-go-round/table-list.png){: .center }

**CHECKLIST**

- ✅ starting a single ScyllaDB instance locally (I'm assuming you have Docker Desktop installed)
- ✅ creating a new Scylla schema with multiple tables
- ☑️ auto-generating the Scylla types for go from this

## Auto-generating the Go Types

Finally! Let's use our Python script:

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round ‹main›
╰─➤  python scylla_go_round/main.py
usage: main.py [-h] --keyspace KEYSPACE [--host HOST] [--username USERNAME] [--password PASSWORD]
               [--output-dir OUTPUT_DIR]
main.py: error: the following arguments are required: --keyspace
```

Oh one other thing is you'll probably want to run your container with its network exposed to the host so you can access it via the scylla-go-round util (at least if you're following this example).

So now let's try our `docker run` portion again:

```
docker run --rm -ti -p 127.0.0.1:9042:9042 --name scyllaU -d scylladb/scylla:5.2.0 --overprovisioned 1 --smp 1 --listen-address 0.0.0.0 --broadcast-rpc-address 127.0.0.1
```

And then locally:

```
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round ‹main›
╰─➤  nc -vz localhost 9042
Connection to localhost port 9042 [tcp/*] succeeded!

(scylla-go-round-py3.12)
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round/scylla_go_round ‹main*›
╰─➤  scyllago --keyspace dog_park --host localhost
Schema file created at output/schema.cql
Go types created at output/dog_park_entities.go

(scylla-go-round-py3.12)
╭─johnlarkin@Larkin-MacBook-Air ~/Documents/coding/scylla-go-round/scylla_go_round ‹main*›
╰─➤  cat output/dog_park_entities.go
package scylladb

import (
	"time"
	"gopkg.in/inf.v0"
	"github.com/google/uuid"
)

type DogActivities struct {
    ActivityID uuid.UUID `cql:"activity_id" cql_pk:"true"`
    ActivityDate time.Time `cql:"activity_date"`
    ActivityDetails string `cql:"activity_details"`
    ActivityType string `cql:"activity_type"`
    DogID uuid.UUID `cql:"dog_id"`
    Tags []string `cql:"tags"`
}

type DogProfiles struct {
    DogID uuid.UUID `cql:"dog_id" cql_pk:"true"`
    Age int `cql:"age"`
    Attributes map[string]string `cql:"attributes"`
    Breed string `cql:"breed"`
    Friends []uuid.UUID `cql:"friends"`
    Name string `cql:"name"`
    VaccinationDates []time.Time `cql:"vaccination_dates"`
    Weight *inf.Dec `cql:"weight"`
}

type GroomingSessions struct {
    SessionID uuid.UUID `cql:"session_id" cql_pk:"true"`
    DogID uuid.UUID `cql:"dog_id"`
    GroomerName string `cql:"groomer_name"`
    Notes string `cql:"notes"`
    Services map[string]string `cql:"services"`
    SessionDate time.Time `cql:"session_date"`
}

type OwnerProfiles struct {
    OwnerID uuid.UUID `cql:"owner_id" cql_pk:"true"`
    ContactInfo map[string]string `cql:"contact_info"`
    DogIds []uuid.UUID `cql:"dog_ids"`
    Name string `cql:"name"`
}
```

Finally!

**CHECKLIST**

- ✅ starting a single ScyllaDB instance locally (I'm assuming you have Docker Desktop installed)
- ✅ creating a new Scylla schema with multiple tables
- ✅ auto-generating the Scylla types for go from this

# Conclusion

🎉 Voila! 🎉 just either change the `output-dir` or copy and paste the file with versioning! It should be as easy as that. If there are issues, definitely feel free to reach out to me or submit a PR to the repo!

[comment]: <> (Bibliography)
[pypi-sgr]: https://pypi.org/project/scylla-go-round/
[scylla]: https://www.scylladb.com/
[go]: https://go.dev/
[goscylla-pt1]: https://university.scylladb.com/courses/using-scylla-drivers/lessons/golang-and-scylla-part-1/
[goscylla-pt2]: https://university.scylladb.com/courses/using-scylla-drivers/lessons/golang-and-scylla-part-2-data-types/
[goscylla-pt3]: https://university.scylladb.com/courses/using-scylla-drivers/lessons/golang-and-scylla-part-3-gocqlx/
[gocqlx]: https://github.com/scylladb/gocqlx
[piwheels-sgr]: https://www.piwheels.org/project/scylla-go-round/
[piwheels]: https://www.piwheels.org/
