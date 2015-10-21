package centroids;

import java.util.ArrayList;

import contrast.Contrast;
import streamDataStructures.Subspace;

public class FullSpaceContrastChecker extends ChangeChecker {

	private Contrast contrastEvaluator;
	private Subspace fullSpace;
	private double lastContrast = 0;
	private double threshold;

	public FullSpaceContrastChecker(int numberOfDimensions, Contrast contrastEvaluator, double threshold) {
		int[] dimensions = new int[numberOfDimensions];
		for (int i = 0; i < numberOfDimensions; i++) {
			dimensions[i] = i;
		}
		fullSpace = new Subspace(dimensions);
		this.contrastEvaluator = contrastEvaluator;
		this.threshold = threshold;
	}

	public void setContrastEvaluator(Contrast contrastEvaluator) {
		this.contrastEvaluator = contrastEvaluator;
	}

	@Override
	public boolean checkForChange(ArrayList<Centroid> centroids) {
		double contrast = contrastEvaluator.evaluateSubspaceContrast(fullSpace);
		double difference = Math.abs(lastContrast - contrast);
		lastContrast = contrast;
		System.out.println("Difference: " + difference);
		if (difference > threshold) {
			return true;
		}
		return false;
	}

}
