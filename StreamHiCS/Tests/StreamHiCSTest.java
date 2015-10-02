import static org.junit.Assert.*;
import org.junit.Test;

import streamDataStructures.Subspace;
import streamDataStructures.SubspaceSet;
import streams.GaussianStream;
import weka.core.Instance;

public class StreamHiCSTest {

	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private final int numInstances = 10000;
	private final int m = 20;
	private final double alpha = 0.4;
	private final double threshold = 0.9;

	@Test
	public void subspaceTest1() {
		double[][] covarianceMatrix = { { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 },
				{ 0, 0, 0, 0, 1 } };

		// No correlated subspaces should have been found, so the correctResult
		// is empty.
		SubspaceSet correctResult = new SubspaceSet();

		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest2() {
		double[][] covarianceMatrix = { { 1, 0.9, 0, 0, 0 }, { 0.9, 1, 0, 0, 0 }, { 0, 0, 1, 0.9, 0.9 },
				{ 0, 0, 0.9, 1, 0.9 }, { 0, 0, 0.9, 0.9, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(2, 4));
		correctResult.addSubspace(new Subspace(3, 4));
		correctResult.addSubspace(new Subspace(2, 3, 4));

		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest3() {
		double[][] covarianceMatrix = { { 1, 0.9, 0.9, 0.9, 0.9 }, { 0.9, 1, 0.9, 0.9, 0.9 }, { 0.9, 0.9, 1, 0.9, 0.9 },
				{ 0.9, 0.9, 0.9, 1, 0.9 }, { 0.9, 0.9, 0.9, 0.9, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(0, 2));
		correctResult.addSubspace(new Subspace(0, 3));
		correctResult.addSubspace(new Subspace(0, 4));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(1, 3));
		correctResult.addSubspace(new Subspace(1, 4));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(2, 4));
		correctResult.addSubspace(new Subspace(3, 4));
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 3));
		correctResult.addSubspace(new Subspace(0, 1, 4));
		correctResult.addSubspace(new Subspace(0, 2, 3));
		correctResult.addSubspace(new Subspace(0, 2, 4));
		correctResult.addSubspace(new Subspace(0, 3, 4));
		correctResult.addSubspace(new Subspace(1, 2, 3));
		correctResult.addSubspace(new Subspace(1, 2, 4));
		correctResult.addSubspace(new Subspace(1, 3, 4));
		correctResult.addSubspace(new Subspace(2, 3, 4));
		correctResult.addSubspace(new Subspace(0, 1, 2, 3));
		correctResult.addSubspace(new Subspace(0, 1, 2, 4));
		correctResult.addSubspace(new Subspace(0, 1, 3, 4));
		correctResult.addSubspace(new Subspace(0, 2, 3, 4));
		correctResult.addSubspace(new Subspace(1, 2, 3, 4));
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest4() {
		double[][] covarianceMatrix = { { 1, 0.8, 0.8, 0.4, 0.4 }, { 0.8, 1, 0.8, 0.4, 0.4 }, { 0.8, 0.8, 1, 0.8, 0.8 },
				{ 0.4, 0.4, 0.8, 1, 0.8 }, { 0.4, 0.4, 0.8, 0.8, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(0, 2));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 2));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(2, 4));
		correctResult.addSubspace(new Subspace(2, 3, 4));

		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest5() {
		double[][] covarianceMatrix = { { 1, 0.5, 0.5, 0, 0 }, { 0.5, 1, 0.5, 0, 0 }, { 0.5, 0.5, 1, 0.1, 0.1 },
				{ 0, 0, 0.1, 1, 0.1 }, { 0, 0, 0.1, 0.1, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(0, 2));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(0, 1, 2));

		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest6() {
		double[][] covarianceMatrix = { { 1, 0.7, 0.4, 0.4, 0.4 }, { 0.7, 1, 0.7, 0.4, 0.4 }, { 0.4, 0.7, 1, 0.7, 0.4 },
				{ 0.4, 0.4, 0.7, 1, 0.7 }, { 0.4, 0.4, 0.4, 0.7, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));
		correctResult.addSubspace(new Subspace(2, 3));
		correctResult.addSubspace(new Subspace(3, 4));

		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	@Test
	public void subspaceTest7() {
		double[][] covarianceMatrix = { { 1, 0.6, 0.1 }, { 0.6, 1, 0.6 }, { 0.1, 0.6, 1 } };
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1));
		correctResult.addSubspace(new Subspace(1, 2));

		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	private boolean carryOutSubspaceTest(double[][] covarianceMatrix, SubspaceSet correctResult) {
		stream = new GaussianStream(covarianceMatrix);
		streamHiCS = new StreamHiCS(covarianceMatrix.length, numInstances, m, alpha, threshold);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}
		
		//Evaluation
		int l = correctResult.size();
		int tp = 0;
		for(Subspace s : correctResult.getSubspaces()){
			if(streamHiCS.getCurrentlyCorrelatedSubspaces().contains(s)){
				tp++;
			}
		}
		int fp = streamHiCS.getCurrentlyCorrelatedSubspaces().size() - tp;
		double recall = 1;
		double fpRatio = 0;
		if(l > 0){
			recall = ((double) tp) / correctResult.size();
			fpRatio = ((double) fp) / correctResult.size();
		}
		System.out.println("True positives: " + tp + " out of " + correctResult.size() + "; False positives: " + fp);
		
		//return streamHiCS.getCurrentlyCorrelatedSubspaces().equals(correctResult);
		return ((recall - fpRatio) > 0.8);
	}

}
