// Agent sample_agent in project pruning

/*
 * All possible beliefs
 * 		remaingLayers. -> I belief that remains layers to prune
 * 		decreasePerformance. -> I belief that pruning decrease the performance
 * 		trainAgain. -> I belief that is necessary to train the model again
 */


/* Initial goals */
!prune.


/* Plans */
+!prune : remaingLayers <- .print("Chama o python"); make_prune; !verifyPerformance.


+!verifyPerformance <- .print("Reavaliando modelo"); verify; !decide.


+!decide: not remaingLayers <- .print("O processo acabou"); just_end.


+!decide: decreasePerformance & trainAgain <- .print("Treine novamente"); train; !prune.


+!decide: decreasePerformance & not trainAgain <- .print("Aumente a performance"); undo_prune; !prune.


+!decide: not decreasePerformance <- .print("Continue com o pruning"); continue_pruning; !prune.