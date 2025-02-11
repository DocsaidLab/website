---
slug: fourier-transform
title: A Brief Introduction to Fourier Transform
authors: Z. Yuan
image: /en/img/2024/1202.webp
tags: [fourier-transform, signal-processing]
description: A simple introduction to the basic concepts of Fourier Transform.
---

import InteractiveSineWave from '@site/src/components/InteractiveSineWave';
import WaveSuperposition from '@site/src/components/WaveSuperposition';
import FourierTransformDemo from '@site/src/components/FourierTransformDemo';

When writing my paper notes, I encountered the Fourier Transform and wanted to discuss it, but the topic became too long and started to overshadow the main content.

So I decided to separate this section and provide a simple introduction to the related concepts.

<!-- truncate -->

## Trigonometric Functions

We all learned trigonometric functions when we were younger, and most of us are probably masters of trigonometry.

Trigonometric functions have widespread applications in physics and engineering, particularly in describing waves and vibrations.

A wave is essentially a way of transferring energy, and this transfer often appears in the form of "periodic oscillations."

Periodic?

A common example might be a pendulum, which swings back and forth in a periodic motion. When we observe its motion path, we can see that its displacement changes like a wave, moving from positive values to zero, then to negative values, and then repeating.

So we try to use trigonometric functions to describe the wave more precisely, mathematically represented as:

$$
y(t) = A \sin(2\pi f t)
$$

Each variable here is related to the pendulum's motion:

- $y(t)$: The displacement of the pendulum at time $t$.
- $A$: The amplitude, representing the maximum distance the pendulum swings.
- $f$: The frequency, representing how many times the pendulum swings per second.

With the sine function, we can more accurately describe the characteristics of the wave.

If you look closely, you will notice that some waves do not start at zero but rather start oscillating after a certain time offset, which is represented by the "phase $\phi$."

The complete waveform formula can be written as:

$$
y(t) = A \sin(2\pi f t + \phi)
$$

Or it can be expressed using the cosine function:

$$
y(t) = A \cos(2\pi f t + \phi)
$$

The difference between sine and cosine waves is in the phase shift, and both can be used to describe wave variations.

Additionally, to describe the rate of change of the wave more effectively, we can use the "angular velocity $\omega$," which represents the rate of phase change per second, with the formula:

$$
\omega = 2\pi f
$$

For easier observation, I have created an interactive chart where you can adjust the amplitude, frequency, and phase to observe the changes in the sine wave.

<InteractiveSineWave />

### Superposition of Waves

With sine and cosine waves, it's like having the basic elements of the x-axis and y-axis; we can combine different waveforms to form more complex waves, such as:

$$
y(t) = A \cos(2\pi f t) + B \sin(2\pi f t)
$$

In this equation, $A$ and $B$ are coefficients that determine the size of the cosine and sine components. By superimposing multiple waves, we can form complex waveforms.

You can use the following interactive chart to adjust the parameters of two sine waves and observe the waveform formed by their superposition.

<WaveSuperposition />

### Complex Form

While the trigonometric representation of sine and cosine waves is very intuitive, using complex numbers to represent waveforms is more concise and powerful in mathematical and engineering applications.

Why?

Because complex numbers can unify the combination of sine and cosine into a single formula and make it easier to handle operations like superposition, differentiation, and integration.

Before understanding the complex form, let's quickly review the basic concept of complex numbers. A complex number consists of a real part and an imaginary part, written as:

$$
z = a + bi
$$

Where:

- $a$ is the real part.
- $b$ is the imaginary part.
- $i$ is the imaginary unit, satisfying $i^2 = -1$.

Complex numbers can also be expressed in "polar form" as:

$$
z = r(\cos\theta + i\sin\theta)
$$

Here:

- $r = \sqrt{a^2 + b^2}$ is the modulus of the complex number, representing the distance from the origin to the point $(a, b)$.
- $\theta = \tan^{-1}(b/a)$ is the argument of the complex number, representing the angle of rotation from the positive real axis.
- $\cos\theta$ and $\sin\theta$ define the direction corresponding to the angle $\theta$.

In Cartesian coordinates, the point $(a, b)$ can be converted to polar coordinates $(r, \theta)$.

Substituting $r$ and $\theta$ into the trigonometric expression, we get $z = r(\cos\theta + i\sin\theta)$.

## Euler's Formula

From a mathematical perspective, the exponential function $e^x$ can be expanded into a power series:

$$
e^x = 1 + \frac{x}{1!} + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots
$$

This series is intuitive when $x$ is a real number, but it also applies when $x$ is a complex number.

For example, when $x = i\theta$, we substitute into the formula to get:

$$
e^{i\theta} = 1 + \frac{i\theta}{1!} + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^3}{3!} + \cdots
$$

Expanding $(i\theta)^n$, and noting the periodicity of powers of $i$ ($i^2 = -1, i^3 = -i, i^4 = 1$), we can separate the real and imaginary parts of the expansion:

The real part is:

$$
1 + \frac{(i\theta)^2}{2!} + \frac{(i\theta)^4}{4!} + \cdots = 1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots
$$

This is exactly the expansion of $\cos\theta$, and similarly, the imaginary part is:

$$
\frac{(i\theta)}{1!} + \frac{(i\theta)^3}{3!} + \frac{(i\theta)^5}{5!} + \cdots = i\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)
$$

This is exactly the expansion of $\sin\theta$.

Therefore, combining the real and imaginary parts, we obtain the famous Euler's formula:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

Here:

- $e^{i\theta}$ is the exponential representation of the complex number, containing the real part $\cos\theta$ and the imaginary part $i\sin\theta$.
- The real and imaginary parts of the complex number correspond to the horizontal and vertical movements of the waveform.

## Complex Waves

<div align="center">
<figure style={{"width": "40%"}}>
![ComplexWave](./img/Euler's_formula.png)
<figcaption>Geometric meaning of Euler's formula, source: [**Wikipedia**](https://en.wikipedia.org/wiki/Euler%27s_formula)</figcaption>
</figure>
</div>

---

We can use Euler's formula to rewrite the waveform equation. For example, consider the following complex waveform:

$$
z(t) = A e^{i(2\pi f t + \phi)}
$$

Expanding $e^{i(2\pi f t + \phi)}$ and using Euler's formula, we get:

$$
z(t) = A \left[\cos(2\pi f t + \phi) + i\sin(2\pi f t + \phi)\right]
$$

This represents the same waveform having both:

- **Real part** $\Re(z) = A \cos(2\pi f t + \phi)$
- **Imaginary part** $\Im(z) = A \sin(2\pi f t + \phi)$

Thus, if we want to represent sine and cosine waves using complex numbers, we can use the real and imaginary parts of the complex number to represent them:

1. **Sine wave** $y(t) = A \sin(2\pi f t + \phi)$:

   - We can represent it using the imaginary part of the complex number:
     $$
     y(t) = \Im \left(A e^{i(2\pi f t + \phi)}\right)
     $$

2. **Cosine wave** $y(t) = A \cos(2\pi f t + \phi)$:
   - We can represent it using the real part of the complex number:
     $$
     y(t) = \Re \left(A e^{i(2\pi f t + \phi)}\right)
     $$

Here, $A e^{i(2\pi f t + \phi)}$ is the unified complex representation, where the real and imaginary parts correspond to the cosine and sine waves, respectively.

### Superposition of Waves

If we directly use trigonometric functions to represent the superposition of two waveforms, we need to use the addition formulas for trigonometric functions, such as:

$$
\cos(A + B) = \cos A \cos B - \sin A \sin B
$$

and

$$
\sin(A + B) = \sin A \cos B + \cos A \sin B
$$

When superimposing two waveforms:

$$
y_1(t) = A_1 \cos(2\pi f_1 t + \phi_1), \quad y_2(t) = A_2 \sin(2\pi f_2 t + \phi_2)
$$

We would need to expand each waveform separately, apply the cosine and sine addition formulas, and then organize the real and imaginary parts.

This process can be tedious and prone to errors, especially when more waveforms are involved, as the complexity of the formulas increases exponentially.

:::tip
Aside from high school students in Taiwan, I doubt anyone would want to manually calculate this, right?
:::

In the complex form, each waveform is represented as:

$$
z_1 = A_1 e^{i(2\pi f_1 t + \phi_1)}, \quad z_2 = A_2 e^{i(2\pi f_2 t + \phi_2)}
$$

The superposed waveform becomes:

$$
z(t) = z_1 + z_2 = A_1 e^{i(2\pi f_1 t + \phi_1)} + A_2 e^{i(2\pi f_2 t + \phi_2)}
$$

The key here is that the addition of complex numbers is "linear," meaning the magnitudes and phases of the two waveforms can be directly added or kept separate without needing to expand the trigonometric formulas.

On the other hand, complex numbers inherently contain both real and imaginary parts, which already carry the information for the cosine and sine components of the waveform. Therefore, there is no need to manually handle the decomposition of trigonometric functions.

Another advantage of using complex numbers is that they allow us to clearly separate the "amplitude" and "phase" of the waveform:

- The amplitude is determined by the magnitude $|z|$.
- The phase is determined by the argument $\arg(z)$.

When superimposing waveforms, these pieces of information can be handled separately or together in calculations, without needing to expand into trigonometric sum or difference formulas. For example, if we want to simply analyze the amplitude of the superposed waveform, we can directly calculate $|z| = \sqrt{\Re(z)^2 + \Im(z)^2}$.

Finally, phase rotation of the waveform corresponds to the shift of the complex number's argument. If we need to adjust the phase of the superposed waveform, we only need to add a rotation angle $\Delta\theta$ to $e^{i\theta}$, without separately adjusting the phases of the cosine and sine components.

## Fourier Transform

All that has been discussed so far serves to help us understand the mathematical formula of the Fourier Transform.

The core concept of the Fourier Transform is:

- **Any signal can be represented as a sum of sine and cosine waves**.

This means that even if a signal looks very complex, such as a piece of music, an image, or a pulse, we can still decompose it into simple, basic waveforms. These basic waveforms are the familiar sine and cosine waves, and they combine in different frequencies, amplitudes, and phases to form the complete signal.

Sine and cosine waves have powerful mathematical properties. For any periodic phenomenon, sine and cosine waves can be viewed as a set of "basis functions," just like the $x$, $y$, and $z$ coordinate axes are used to describe positions in space. By appropriately combining them, we can represent any complex shape or variation.

Suppose a signal $y(t)$ is a time-varying function. The Fourier Transform helps us answer two key questions:

1. **What frequencies exist in this signal?**
2. **What are the amplitude and phase of each frequency?**

This frequency decomposition allows us to understand the signal from a completely different perspective. Instead of directly observing the waveform changing over time, we can see the signal's spectral characteristics more clearly.

The mathematical definition of the Fourier Transform provides the method for converting a signal from the "time domain" to the "frequency domain," with the core formula being:

$$
Y(f) = \int_{-\infty}^\infty y(t) e^{-i 2\pi f t} \, dt
$$

The physical meaning of each part of the formula is as follows:

1. **Original Signal $y(t)$**:

   $y(t)$ is a function defined in the time domain. This can be any form of signal, such as the sound wave of a piece of music, the amplitude variations of an electrical signal, or a pulse sequence that jumps within a time interval.

   $y(t)$ contains the amplitude values of the signal at each moment of time $t$.

2. **Complex Exponential Function $e^{-i 2\pi f t}$**:

   This is the key part of the Fourier Transform, and it is actually a combination of sine and cosine waves:

   $$
   e^{-i 2\pi f t} = \cos(2\pi f t) - i \sin(2\pi f t)
   $$

   In the Fourier Transform, $e^{-i 2\pi f t}$ is used to match a complex waveform with frequency $f$ to $y(t)$, extracting the strength of the signal at that frequency.

---

The integral operation is essentially an inner product calculation, used to measure the similarity between $y(t)$ and $e^{-i 2\pi f t}$.

This "similarity" determines the contribution of the signal at frequency $f$.

The Fourier Transform scans over all possible frequencies $f$.

For each $f$, the value $Y(f)$ calculated by the integral gives the strength at that frequency, which is why $Y(f)$ is called the "spectrum."

Instead of talking more about it, let's try calculating it ourselves:

Consider a signal with a single frequency:

$$
y(t) = A \cos(2\pi f_0 t + \phi)
$$

Using Euler's formula, we express it as:

$$
y(t) = \Re\left\{ A e^{i (2\pi f_0 t + \phi)} \right\}
$$

Now, performing the Fourier Transform:

$$
Y(f) = \int_{-\infty}^\infty y(t) e^{-i 2\pi f t} \, dt
$$

Substituting the expression for $y(t)$:

$$
Y(f) = \int_{-\infty}^\infty \Re\left\{ A e^{i (2\pi f_0 t + \phi)} \right\} e^{-i 2\pi f t} \, dt
$$

Since the core of the Fourier Transform is the integral operation, its effect is to match every point in the time domain with a frequency waveform. If $f = f_0$, the $e^{-i 2\pi f t}$ and $e^{i 2\pi f_0 t}$ align perfectly, and the result of the integral will be non-zero.

If $f \neq f_0$, these two waveforms do not match in phase, and the integral result approaches zero.

When the frequency matches, we get:

$$
Y(f_0) = A e^{i \phi}
$$

This shows that the spectrum $Y(f)$ contains not only the strength of the frequency (given by $A$) but also the phase of the frequency (given by $\phi$).

This is the reason why the Fourier Transform can fully describe a signal.

## Fourier Series

The Fourier Series is a special case of the Fourier Transform, mainly used for periodic signals.

Unlike the Fourier Transform, which extends the signal's spectrum to a continuous range of frequencies, the Fourier Series focuses on using a set of "discrete frequencies" to describe a periodic signal.

The core concept of the Fourier Series is:

- **Any periodic signal can be represented as a linear combination of sine and cosine waves at discrete frequencies**.

This means that as long as the signal is periodic, we can approximate it using a finite number of sine and cosine waves, and as the frequency combinations increase, the approximation becomes more accurate.

Suppose a periodic signal $x(t)$ with period $T$ can be represented by the Fourier Series as:

$$
x(t) = a_0 + \sum_{n=1}^\infty \left[ a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right) \right]
$$

Here, $a_0, a_n, b_n$ are the coefficients of the Fourier Series, representing the amplitudes of the sine and cosine waves at different frequencies, and these coefficients are determined by the following formulas:

1. **DC Component ($a_0$)**:

   $$
   a_0 = \frac{1}{T} \int_{0}^T x(t) \, dt
   $$

   It represents the average value of the signal, or the DC component over one period.

2. **Cosine Coefficients ($a_n$)**:

   $$
   a_n = \frac{2}{T} \int_{0}^T x(t) \cos\left(\frac{2\pi n t}{T}\right) \, dt
   $$

   It represents the amplitude of the cosine wave at frequency $\frac{n}{T}$.

3. **Sine Coefficients ($b_n$)**:

   $$
   b_n = \frac{2}{T} \int_{0}^T x(t) \sin\left(\frac{2\pi n t}{T}\right) \, dt
   $$

   It represents the amplitude of the sine wave at frequency $\frac{n}{T}$.

Similarly, to simplify the representation and computation, the Fourier Series is often written in complex form:

$$
x(t) = \sum_{n=-\infty}^\infty c_n e^{i \frac{2\pi n t}{T}}
$$

Where $c_n$ are the complex coefficients, defined as:

$$
c_n = \frac{1}{T} \int_{0}^T x(t) e^{-i \frac{2\pi n t}{T}} \, dt
$$

In this form:

- The real part $\Re\{c_n\}$ corresponds to $a_n$.
- The imaginary part $\Im\{c_n\}$ corresponds to $b_n$.

The Fourier Series has wide applications in engineering and science, such as studying periodic vibrations in mechanical structures and analyzing periodic electrical signals, like square waves, triangle waves, and sawtooth waves.

## Conclusion

Here, we can only briefly introduce the relevant concepts of the Fourier Transform.

There is more to explore, such as the Discrete Fourier Transform (DFT), Fast Fourier Transform (FFT), the relationship between Fourier Transform and convolution, frequency-domain filter design and implementation, and the extensions of Fourier analysis in quantum physics and image processing, which could take days to fully discuss.

Interested readers can consult further resources to learn more about the principles and applications of Fourier Transform.

With the remaining space, I'll add a fun interactive activity at the end to experience the charm of Fourier!

## Interactive Activity

Based on the settings, the program will plot the corresponding waveform in the time domain and show its analysis results in the frequency domain.

You will see that the frequency domain distribution matches the values on the "settings panel," with other frequencies generated to approximate the waveform.

<FourierTransformDemo />
