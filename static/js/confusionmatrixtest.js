// busbus: 7344 busper:  342 buspre 2304 perbus 3487 
// perper 965 perpre 2656 prebus 5687 preper 659 prepre 3212        
        
        var confusionMatrix = [
            [7344, 342, 2304],
            [3487, 965, 2656],
            [5687, 659, 3212]
		];

        var bb = confusionMatrix[0][0];
        var bpe = confusionMatrix[0][1];
        var bpr = confusionMatrix[0][2];
        var peb = confusionMatrix[1][0];
        var pepe = confusionMatrix[1][1];
        var pepr = confusionMatrix[1][];
        var prb = confusionMatrix[2][0];
        var prpe = confusionMatrix[2][1];
        var prpr = confusionMatrix[2][2];

        var p = bb + pepe + prpr;
        var n = bpe + bpr + peb + pepr + prb +prpe;

        var accuracy = (p)/(p+n);
        // var f1 = 2*tp/(2*tp+fp+fn);
        // var precision = tp/(tp+fp);
        // var recall = tp/(tp+fn);

        // accuracy = Math.round(accuracy * 100) / 100
        // f1 = Math.round(f1 * 100) / 100
        // precision = Math.round(precision * 100) / 100
        // recall = Math.round(recall * 100) / 100

        // var computedData = [];
        // computedData.push({"F1":f1, "PRECISION":precision,"RECALL":recall,"ACCURACY":accuracy});

        // var labels = ['Class A', 'Class B'];
		// Matrix({
		// 	container : '#container',
		// 	data      : confusionMatrix,
		// 	labels    : labels,
        //     start_color : '#ffffff',
        //     end_color : '#e67e22'
		// });

		// // rendering the table
        //  var table = tabulate(computedData, ["F1", "PRECISION","RECALL","ACCURACY"]);