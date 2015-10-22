package centroids;

import java.util.ArrayList;

import contrast.Contrast;
import streamDataStructures.Subspace;

public class FullSpaceContrastChecker extends ChangeChecker {

	private Contrast contrastEvaluator;
	private Subspace fullSpace;
	private double lastContrast = 0;
	private double minContrast = Double.MIN_VALUE;
	private double maxContrast = Double.MIN_VALUE;
	private double threshold;
	private boolean init = false;

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

		System.out.println("Contrast: " + contrast);

		if(!init){
			minContrast = contrast;
			maxContrast = contrast;
			init = true;
		}
		
		double minDifference = minContrast - contrast;
		double maxDifference = contrast - maxContrast;

		if (contrast < minContrast) {
			minContrast = contrast;
		}
		if (contrast > maxContrast) {
			maxContrast = contrast;
		}

		// double difference = Math.abs(lastContrast - contrast);
		// lastContrast = contrast;
		System.out.println("MinDifference: " + minDifference + ", MaxDifference: " + maxDifference);
		if (Math.abs(minDifference) > threshold || Math.abs(maxDifference) > threshold) {
			minContrast = 0;
			maxContrast = 0;
			init = false;
			return true;
		}
		return false;
	}

}
