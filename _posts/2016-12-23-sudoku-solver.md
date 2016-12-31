---
title: Backtracking Sudoku Solver
layout: post
---

Bored of solving sudokus normally? Try out this sudoku solver. 

![Alt text][backtrackgif]{: .center-image }

Above, is a visualization from Wikipedia, about the sudoku backtracking algorithm.

Concept
=======
After waiting on the airplane and solving another soduko from the back of a Skymiles magazine, I thought it might be fun to build a sudoku solver. At first, I was planning on implementing essentially a brute force approach, where I would try a possible combination for a cell, and then with that possible cell filled, try to fill out the cell from there. I figured this would be considered brute force.

As it turns out, that approach is a little bit more sophisticated then just basic brute force, and is actually commonly called [backtracking][backtrack]. You can read more about it [below](#theory). The algorithm makes pretty intuitive sense. You find the possible options for a given cell, and then recurse on the given sudoku from there. When you find one that is fully filled and satisfied, you just short out, and keep returning true and then you've got yourself the golden sudoku. The solution! If you're just interested in grabbing the actual program, [you can find it here][code].

The actual textbook definition of brute-force means to iterate over the entire search space, and check whether that solution is valid. So you can imagine that this would be **literally** generating all possible values for all open cells. This is also going to be discussed further [below](#runtime). 

Program
=======
The program is pretty simple. There is a menu when you first start the program where you can enter in sudoku matrices either from another file, from `stdin`, solve a default sudoku, or cycle through a directory. The only other thing to know is that the unfilled spaces on the sudoku are marked by EITHER 0 or -1. You can choose at your fancy. Let me know if there are any issues besides that. 

Review of Sudoku
================
I figured I should throw this here in case some people aren't totally clear of the rules of sudoku or even what it is. Sudoku is a grid of 9 x 9 cells. That's 81 cells in total. There are 9 different unique subblocks, with have 9 cells in them. Some of these cells might be filled at the start of the puzzle. There *has* to be one unique solution to every sudoku puzzle, or else it is not valid. The rules in terms of filling the cells are found below:

1. Each block must contain the numbers 1-9
2. Each row must contain the numbers 1-9
3. Each column must contain the numbers 1-9

And that's really it! Otherwise then that, it's just filling in the cells. 
 
Theory behind Backtracking {#theory}
==========================
Ok so the theory behind backtracking. How is this different than the brute force approach? This [site][btlink1] has a pretty good synopsis. One that I probably wish I would have read before just blindly leaping into trying to solve this problem. As the author puts it, recursion is really the name of the game. The idea is that we pick one possible starting option out of a bunch and then we recurse on that solution. If that solution doesn't work, then we go back to our original assertion, and we try another. Then we recurse off of that as well. If none of the starting options actually work, then there are no solutions to the problem itself. 
The crux of backtracking is also that when you recurse, you only consider the solutions or guesses that you have already made.  

You can also think about this as a graph search problem. Wikipedia provides an interesting way of generalizing the backtracking algorithm.

Note, in the following, *P* is essentially data that is being taken in and is related to the problem. 
1. *root*(*P*): return the partial candidate at the root of the search tree. 
2. *reject*(*P*,*c*): return *true* only if the partial candidate *c* is not worth completing.
3. *accept*(*P*,*c*): return true if *c* is a solution of *P*, and false otherwise.
4. *first*(*P*,*c*): generate the first extension of candidate *c*.
5. *next*(*P*,*s*): generate the next alternative extension of a candidate, after the extension *s*.
6. *output*(*P*,*c*): use the solution *c* of *P*, as appropriate to the application.

```python
def bt(c):

    if reject(P,c) then return
    if accept(P,c) then output(P,c)
    s ← first(P,c)
    while s ≠ Λ do
        bt(s)
        s ← next(P,s)
```

Asymptotic Runtime {#runtime}
==================
So how does backtracking differ in runtime than the real brute force, just try every number in every open spot? Well let's break down each individual runtime. 

### Brute Force Algorithm
If we think about having a normal sudoku matrix, with some $$ m $$ blank cells, and $$ n $$ possible options for each cell (in this case, for normal sudoku it's just 9), we can think about the brute force runtime being:

$$ 
O(n ^ m ) = O(9 ^ m) 
$$

for our normal sudoku game. With some variations, there might be more than just nine possibilities for each cell.

Ok so, that's exponential. Which is obviously not great. What about backtracking?

### Backtracking Algorithm 
Before we make any claims, let's think about the worst case for a second. In the worst case, backtracking **could still fail and just revert to a normal bruteforce algorithm**!! The concept of backtracking really hinges on the idea that we are able to guess a good starting value for our initial cell. But let's think about it. Let's say that we find our cell that we're going to fill and the correct value for the cell is actually 9, but we're going to go through and try 1, 2, 3, ... 7, 8 all before getting to 9. For each one of those, we recurse on our guess and try all the possibilities. 

So you can imagine in the **very** worst case, that we get really really unlucky and that backtracking still has to fully exhaust our search space. That being said, the expected runtime for our backtracking algorithm is going to be much better in reality. We are most likely going to find a good starting point and then solve the entire matrix, much more quickly than with brute force. 

But still... I have to report the worst case runtime as:

$$
O(n^m ) = O(9^m)
$$

Real Runtime
============

I figured I would compute the runtime of a few sudoku's. These are the ones that came installed in the directory. I just pulled them off [this website][sudokuwebsite]. Mad props to them for all the free sudokus with the solutions.

Here are some of the runtimes. Note, I am just reporting the time it took to solve the matrix. I did this using python's `time` library. Specifically, this snippet of code:

~~~python
t0 = time.time()
print("Attempting to solve sudoku from filename: {}".format(filename))
solved = solve_sudoku(matrix)
if solved == None:
    print "There is no solution"
else:
    print("Solved sudoku with filename: {}".format(filename))
    t1 = time.time()
    print("Time to solve sudoku: {}".format(t1-t0))
    solution_dict[filename[7:10]] = solved
~~~
 
Here are the runtimes:

| Sudoku Name    | User Time (sec) |
| :------------- | ---------------:|
| <span style="font-family: Consolas;"> sudoku_257.txt </span> | 21.5058   |
| <span style="font-family: Consolas;"> sudoku_258.txt </span> | 3.15790   |
| <span style="font-family: Consolas;"> sudoku_259.txt </span> | 0.07309   |
| <span style="font-family: Consolas;"> sudoku_260.txt </span> | 2.63807   |
| <span style="font-family: Consolas;"> sudoku_261.txt </span> | 4.79477   |
| <span style="font-family: Consolas;"> sudoku_262.txt </span> | 1.11362   |


You can see that most of them were solved with ease, but `sudoku_257.txt` the program really chomped on. I'm not totally sure why, but if I were to guess, the backtracking algorithm just didn't work super great with this one, and it might have reverted to something similar to the brute force algorithm. Regardless, it's cool to have these variations in the actual program.


Conclusion
==========
I actually struggled a bit with writing the code. It took me awhile to figure out how to cover all my base cases. I realized that this is on Geeks for Geeks, or at least I'm pretty sure, but I wasn't really about looking at their solution, and I coded mine up from scratch. It's not the most elequent, but I'm pretty proud of it. I know it could be much prettier, but at the moment, I'm content. Anyway, [check out the program here!][code]. It's got some nice features, such as being able to read in a matrix from stdin, from a file, default, and cycle through a directory and check the inputs. Let me know if anyone has any improvements! The code isn't fully optimized yet, just a solution, that works and that I'm happy and proud with. 
Overall, it was a solid project to work on. Definitely awesome to pick up a new concept like backtracking, as this was my first taste of it in both academics and actual programming. I'm looking forward to learning more about it. As always, definitely feel free to comment or shoot me an email if there are any questions, or anything that I can do a lot better. Always looking to improve. 

[comment]: <> (Bibliography)
[code]: https://github.com/johnlarkin1/sudoku-solver
[backtrackgif]: https://upload.wikimedia.org/wikipedia/commons/8/8c/Sudoku_solved_by_bactracking.gif
[backtrack]: https://en.wikipedia.org/wiki/Backtracking
[btlink1]: http://algorithms.tutorialhorizon.com/introduction-to-backtracking-programming/
[sudokuwebsite]: http://www.puzzles.ca/sudoku.html
