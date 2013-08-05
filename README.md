assoc-space
===========

Compute association strength over semantic networks in a dimensionality-reduced form.

This code is used for doing cool things with
[ConceptNet](http://conceptnet5.media.mit.edu). It's probably useful for other things too.

The high-level idea is:

- You have a relatively unstructured semantic network, with undirected edges
- You want to find out how strongly connected two nodes of the network are (based on how many paths get you from node 1 to node 2, vs. how complex those paths are)
- One way to do that is to apply spreading activation from node 1, and see how much of it reaches node 2, except that's terribly inefficient
- Dimensionality reduction applies:
  - You could represent this semantic network as a sparse matrix A of which nodes are connected to which other nodes
  - You can represent a reasonable approximation to this semantic network as a smaller dense matrix U and a diagonal Σ, where U · Σ · U^T ~= A
  - (This is an application of SVD)
- You can now simulate spreading activation with a really straightforward operation on Σ.
- That's what *assoc-space* does.

