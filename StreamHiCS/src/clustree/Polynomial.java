package clustree;

/*
Copyright © 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
 */

public class Polynomial {

	protected Polynomial() {
	}

	public static double p1evl(double x, double coef[], int N) throws ArithmeticException {
		double ans;

		ans = x + coef[0];

		for (int i = 1; i < N; i++) {
			ans = ans * x + coef[i];
		}

		return ans;
	}

	public static double polevl(double x, double coef[], int N) throws ArithmeticException {
		double ans;
		ans = coef[0];

		for (int i = 1; i <= N; i++) {
			ans = ans * x + coef[i];
		}

		return ans;
	}
}
