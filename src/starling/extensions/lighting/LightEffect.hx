// =================================================================================================
//
//	Starling Framework
//	Copyright Gamua. All Rights Reserved.
//
//	This program is free software. You can redistribute and/or modify it
//	in accordance with the terms of the accompanying license agreement.
//
// =================================================================================================

package starling.extensions.lighting;

import openfl.display3D.Context3D;
import openfl.display3D.Context3DProgramType;
import openfl.geom.Vector3D;
import openfl.utils.AGALMiniAssembler;
import openfl.utils.ByteArray;
import openfl.utils.Endian;
import openfl.Vector;
import starling.rendering.FilterEffect;
import starling.rendering.MeshEffect;
import starling.rendering.Program;
import starling.rendering.VertexDataFormat;
import starling.textures.Texture;
import starling.utils.Color;
import starling.utils.MathUtil;
import starling.utils.RenderUtil;
import starling.utils.StringUtil;

/** @private
 *
 *  LightEffect builds the shader program used by [`LightStyle`](LightStyle.hx:1) to render
 *  meshes lit by an arbitrary number of [`LightSource`](LightSource.hx:1) instances.
 *
 *  The implementation has two compile-time paths:
 *
 *  - **Non-Flash targets (default, html5 / native / hl / neko / mobile):** the fragment
 *    shader is hand-written in GLSL and embedded in a 0xB0-prefixed ByteArray so that
 *    OpenFL's `Program3D.upload` →  `AGALConverter.convertToGLSL` passes it through
 *    verbatim. The vertex shader is still AGAL (v2). Constant upload uses the standard
 *    `setProgramConstantsFromVector` calls — the linked GL program exposes its
 *    `fcN` uniforms by name, which OpenFL's `__buildAGALUniformList` wires to the
 *    same `__fragmentConstants` array regardless of source. This lifts the per-shader
 *    light cap from "AGAL register exhaustion" to "GPU uniform budget".
 *
 *  - **Flash target (compile-time fallback via `#if flash`):** both shaders are
 *    AGAL v2. The fragment shader supports up to 27 lights, derived from
 *    `(fc63 - fc10) / 2`. Flash cannot use the GLSL path because OpenFL's GLSL
 *    machinery is gated on `#if !flash`.
 *
 *  The `programVariantName` getter uses a dual-schema encoding so the 32-bit
 *  `UInt` variant key stays collision-free for arbitrary light counts (see the
 *  comment on `get_programVariantName` for the bit layout).
 */
@:keep
class LightEffect extends MeshEffect
{
	public var numLights(get, set):Int;
	public var cameraPosition(get, set):Vector3D;
	public var normalTexture(get, set):Texture;

	public static var VERTEX_FORMAT:VertexDataFormat =
		MeshEffect.VERTEX_FORMAT.extend(
			"normalTexCoords:float2, material:bytes4, xAxis:float2, yAxis:float2, zScale:float1"
		);

	// ─────────────────────────────────────────────────────────────────────────
	// Private configuration constants
	// ─────────────────────────────────────────────────────────────────────────

	/** First fragment-constant register used to upload light data.
	 *  Light `i` occupies `fc(FC_LIGHT_BASE + 2*i)` (position/direction) and
	 *  `fc(FC_LIGHT_BASE + 2*i + 1)` (color). */
	private static inline var FC_LIGHT_BASE:Int = 10;

	/** Number of fc-register slots reserved before light data
	 *  (`fc0` = constants vec4, `fc3` = camera vec4, plus headroom).
	 *  Used in the GLSL-path runtime cap calculation. */
	private static inline var FC_RESERVED:Int = 10;

	/** AGAL version we compile the vertex shader (and, on Flash, the fragment shader)
	 *  as. v2 lifts the fragment-constant ceiling from `fc27` to `fc63`. Universally
	 *  supported by every shipping OpenFL backend. */
	private static inline var AGAL_VERSION:Int = 2;

	/** Compile-time switch between the two fragment-shader paths.
	 *  - On non-Flash targets: `true` → GLSL fragment shader via 0xB0 magic byte,
	 *    GPU-bound light cap (typically 64).
	 *  - On Flash: `false` → AGAL v2 fragment shader, light cap of 27.
	 *  Haxe dead-code-eliminates the unused branch at each compile. */
	private static inline var USE_GLSL_FRAGMENT:Bool = #if flash false #else true #end;

	/** Hard ceiling on AGAL-fallback fragment-constant indices. v2 allows
	 *  `fc0`..`fc63`, so lights occupy at most `(64 - FC_LIGHT_BASE) / 2 = 27` slots. */
	private static inline var AGAL_HARD_LIMIT:Int = 27;

	/** Lazily-initialised real ceiling. -1 sentinel = uninitialised; first call to
	 *  `getHardLimit()` resolves the value once and caches it. */
	private static var sHardLimit:Int = -1;

	// ─────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────

	private var _lights:Array<Light>;
	private var _normalTexture:Texture;
	private var _cameraPosition:Vector3D;

	private static var sVector:Vector<Float> = new Vector<Float>(4, true);

	#if !flash
	/** Re-usable AGAL assembler for the vertex shader on the GLSL path. */
	private static var sAssembler:AGALMiniAssembler = new AGALMiniAssembler();
	#end

	public function new()
	{
		super();
		_lights = [];
		_cameraPosition = new Vector3D();
	}

	/** Resolves the real per-draw-call light ceiling, given the active target
	 *  and (on GLSL targets) the GPU's reported `GL_MAX_FRAGMENT_UNIFORM_VECTORS`.
	 *  Always clamped to `LightStyle.MAX_NUM_LIGHTS` so that shader-cache key
	 *  space stays bounded and shader compile time stays small. */
	private static function getHardLimit():Int
	{
		if (sHardLimit >= 0) return sHardLimit;

		var limit:Int;

		if (USE_GLSL_FRAGMENT)
		{
			#if lime
			// GL_MAX_FRAGMENT_UNIFORM_VECTORS = 0x8DFD (GLES2/WebGL1).
			// Some desktop GL drivers report MAX_FRAGMENT_UNIFORM_COMPONENTS instead,
			// but every Lime context backend (HTML5WebGL/NativeOpenGL) exposes this
			// pname. Wrapped in try/catch in case a future backend doesn't.
			var raw:Dynamic = null;
			try
			{
				raw = lime.graphics.opengl.GL.getParameter(
					lime.graphics.opengl.GL.MAX_FRAGMENT_UNIFORM_VECTORS
				);
			}
			catch (e:Dynamic) {}
			var maxVec:Int = 0;
			if (raw != null)
			{
				// Result may be Int or Float depending on backend.
				maxVec = Std.int(raw);
			}
			if (maxVec <= 0) maxVec = 224; // conservative fallback (typical WebGL value)
			limit = Std.int((maxVec - FC_RESERVED) / 2);
			#else
			limit = LightStyle.MAX_NUM_LIGHTS;
			#end
		}
		else
		{
			limit = AGAL_HARD_LIMIT;
		}

		// Always clamp to the documented soft cap so cache key space is bounded
		// and so shader compile time stays in the millisecond range.
		if (limit > LightStyle.MAX_NUM_LIGHTS) limit = LightStyle.MAX_NUM_LIGHTS;
		if (limit < 0) limit = 0;

		sHardLimit = limit;
		return sHardLimit;
	}

	override private function createProgram():Program
	{
		var numLights:Int = Std.int(MathUtil.min(_lights.length, getHardLimit()));

		if (USE_GLSL_FRAGMENT)
		{
			#if !flash
			return createProgramGLSL(numLights);
			#else
			return createProgramAGAL(numLights);
			#end
		}
		else
		{
			return createProgramAGAL(numLights);
		}
	}

	// ─────────────────────────────────────────────────────────────────────────
	// GLSL fragment-shader path — non-Flash targets
	// ─────────────────────────────────────────────────────────────────────────

	#if !flash
	/** Builds the program with an AGAL vertex shader and a hand-written GLSL
	 *  fragment shader embedded via the 0xB0 magic byte. */
	private function createProgramGLSL(numLights:Int):Program
	{
		var vsBytes:ByteArray = sAssembler.assemble(
			Context3DProgramType.VERTEX, buildVertexShaderAGAL(), AGAL_VERSION
		);

		var fsBytes:ByteArray = makeGLSLFragmentProgram(
			buildFragmentShaderGLSL(numLights)
		);

		return new Program(vsBytes, fsBytes);
	}

	/** Wraps a GLSL source string in a 0xB0-magic ByteArray so that
	 *  OpenFL's `AGALConverter.convertToGLSL` returns it verbatim
	 *  (see ref/openfl/src/openfl/display3D/_internal/AGALConverter.hx:60-65). */
	private static function makeGLSLFragmentProgram(glsl:String):ByteArray
	{
		var ba:ByteArray = new ByteArray();
		ba.endian = Endian.LITTLE_ENDIAN;
		ba.writeByte(0xB0);
		ba.writeUTF(glsl);
		return ba;
	}

	/** Builds the GLSL fragment shader for the given number of lights.
	 *
	 *  Layout invariants:
	 *  - Varyings `v0..v7` are named to match what AGALConverter would emit
	 *    from our AGAL vertex shader.
	 *  - Uniforms are named `fc0`, `fc3`, `fc10`, `fc11`, … `fcN`. The
	 *    indices match the ones written via `setProgramConstantsFromVector`
	 *    in `beforeDraw`. Program3D.__buildAGALUniformList discovers these by name. */
	private function buildFragmentShaderGLSL(numLights:Int):String
	{
		var sb:StringBuf = new StringBuf();

		// ── Header / precision ──────────────────────────────────────────────
		sb.add("#ifdef GL_FRAGMENT_PRECISION_HIGH\n");
		sb.add("precision highp float;\n");
		sb.add("#else\n");
		sb.add("precision mediump float;\n");
		sb.add("#endif\n");

		// ── Varyings (must match what the AGAL vertex shader emits via AGALConverter) ─
		// v0 = vertex position, v1 = tex coords, v2 = vertex color*alpha,
		// v3 = normal tex coords, v4 = material, v5/v6/v7 = basis matrix rows.
		sb.add("varying vec4 v0;\n");
		sb.add("varying vec4 v1;\n");
		sb.add("varying vec4 v2;\n");
		sb.add("varying vec4 v3;\n");
		sb.add("varying vec4 v4;\n");
		sb.add("varying vec4 v5;\n");
		sb.add("varying vec4 v6;\n");
		sb.add("varying vec4 v7;\n");

		// ── Samplers ────────────────────────────────────────────────────────
		// AGALConverter names sampler uniforms "sampler0", "sampler1", etc.
		// We follow the same convention so the existing setTextureAt(0, ...) /
		// setTextureAt(1, ...) calls bind to the right uniforms.
		sb.add("uniform sampler2D sampler0;\n");
		if (_normalTexture != null) sb.add("uniform sampler2D sampler1;\n");

		// ── Fixed fragment-constant uniforms ────────────────────────────────
		// fc0 holds [0, 1, 2, 0.1] — same constants as the AGAL path.
		// fc3 holds the camera position.
		sb.add("uniform vec4 fc0;\n");
		sb.add("uniform vec4 fc3;\n");

		// ── Per-light fragment-constant uniforms ────────────────────────────
		// Two vec4s per light: position/direction and color.
		for (i in 0...numLights)
		{
			sb.add("uniform vec4 fc"); sb.add(FC_LIGHT_BASE + 2 * i); sb.add(";\n");
			sb.add("uniform vec4 fc"); sb.add(FC_LIGHT_BASE + 1 + 2 * i); sb.add(";\n");
		}

		// ── main() ──────────────────────────────────────────────────────────
		sb.add("void main()\n{\n");

		// Surface color = texel × vertex tint
		sb.add("\tvec4 surfaceColor = texture2D(sampler0, v1.xy) * v2;\n");

		// Build the per-fragment normal vector in local coordinates.
		if (_normalTexture != null)
		{
			// Sample the normal map and remap from [0,1] to [-1,1], then flip y/z
			// to match the original AGAL shader's convention.
			sb.add("\tvec3 N = texture2D(sampler1, v3.xy).xyz;\n");
			sb.add("\tN.xy = N.xy * fc0.zz - fc0.yy;\n"); // N.xy *= 2; N.xy -= 1
			sb.add("\tN.z = -N.z;\n");
			sb.add("\tN.y = -N.y;\n");
		}
		else
		{
			// Default normal points "into" the screen.
			sb.add("\tvec3 N = vec3(0.0, 0.0, -1.0);\n");
		}

		// Transform the normal into the mesh's local coordinate system using
		// the basis matrix encoded in varyings v5/v6/v7. The original AGAL
		// shader uses an m33 op which AGALConverter implements as
		//   result = N * mat3(v5, v6, v7)
		// (i.e. row-vector × matrix). We replicate that here.
		sb.add("\tmat3 basis = mat3(v5.xyz, v6.xyz, v7.xyz);\n");
		sb.add("\tN = normalize(N * basis);\n");

		// Initialize the accumulator. The original shader uses a `mov` for the
		// first light and `add`s for the rest; here we just zero-init and `+=`.
		sb.add("\tvec4 accum = vec4(0.0);\n");

		// View vector V = camera - position, used by every non-ambient light.
		// Computed once outside the loop for efficiency.
		var anyLit:Bool = false;
		for (i in 0...numLights)
		{
			if (_lights[i].type != LightSource.TYPE_AMBIENT) { anyLit = true; break; }
		}
		if (anyLit)
		{
			// Apply the same 0.1× scaling used in the AGAL shader to keep
			// intermediate vector magnitudes well within mediump-fp range
			// before normalization.
			sb.add("\tvec3 V = normalize((fc3.xyz - v0.xyz) * fc0.w);\n");
		}

		// One block per light. The type-switch is resolved at codegen time so
		// the GLSL compiler sees straight-line code — same compilation cost
		// profile as the AGAL path.
		for (i in 0...numLights)
		{
			var light:Light = _lights[i];
			var lPos:String = "fc" + (FC_LIGHT_BASE + 2 * i);
			var lCol:String = "fc" + (FC_LIGHT_BASE + 1 + 2 * i);

			sb.add("\t{ // light "); sb.add(i); sb.add("\n");

			if (light.type == LightSource.TYPE_AMBIENT)
			{
				// illumination = surface * lightColor * ambientRatio
				sb.add("\t\taccum += surfaceColor * "); sb.add(lCol); sb.add(" * v4.x;\n");
			}
			else
			{
				// L = normalized light vector (from-position for point; direction for directional)
				if (light.type == LightSource.TYPE_POINT)
				{
					sb.add("\t\tvec3 L = normalize(("); sb.add(lPos); sb.add(".xyz - v0.xyz) * fc0.w);\n");
				}
				else // TYPE_DIRECTIONAL
				{
					sb.add("\t\tvec3 L = normalize("); sb.add(lPos); sb.add(".xyz * fc0.w);\n");
				}

				// Phong reflection model.
				sb.add("\t\tfloat LdotN = clamp(dot(L, N), 0.0, 1.0);\n");
				sb.add("\t\tvec3 R = LdotN * fc0.z * N - L;\n");           // R = 2(L.N)N - L
				sb.add("\t\tfloat RdotV = clamp(dot(R, V), 0.0, 1.0);\n");

				// Diffuse: LdotN * lightColor * diffuseRatio (v4.y)
				sb.add("\t\tvec4 diff = LdotN * "); sb.add(lCol); sb.add(" * v4.y;\n");

				// Specular: RdotV^shininess * lightColor * specularRatio (v4.z),
				// pre-multiplied by alpha (surfaceColor.w).
				sb.add("\t\tvec4 spec = pow(RdotV, v4.w) * "); sb.add(lCol); sb.add(" * v4.z * surfaceColor.w;\n");

				// Total illumination from this light:
				//   illumination = surface * diffuse + specular
				sb.add("\t\taccum += surfaceColor * diff + spec;\n");
			}

			sb.add("\t}\n");
		}

		// Restore the surface alpha (lighting math zeroes/over-writes .w).
		sb.add("\taccum.w = surfaceColor.w;\n");
		sb.add("\tgl_FragColor = accum;\n");
		sb.add("}\n");

		return sb.toString();
	}
	#end

	// ─────────────────────────────────────────────────────────────────────────
	// AGAL fragment-shader path — Flash target and `USE_GLSL_FRAGMENT = false` fallback
	// ─────────────────────────────────────────────────────────────────────────

	/** Builds the program with both shaders as AGAL v2.
	 *  This is the fallback used on Flash, where OpenFL's GLSL machinery
	 *  is `#if !flash`-gated. The shader logic is the original AGAL one,
	 *  just with the loop bound widened to allow up to 27 lights. */
	private function createProgramAGAL(numLights:Int):Program
	{
		/** Stage3D uses medium precision in the fp, guaranteeing a range of +/- 2^14.
		 *  As part of the vector normalization, the coordinates need to be squared, and
		 *  that easily overshoots those bounds. To be on the safe side, the vector is
		 *  thus scaled to 10% of its original length before normalizing. */

		var nrm:String -> String = function(register:String):String
		{
			return StringUtil.format(
				"mul {0}.xyz, {0}.xyz, fc0.www \n" +
				"nrm {0}.xyz, {0}.xyz", [register]
			);
		}

		var fragmentShader:Array<String> = [
			FilterEffect.tex("ft0", "v1", 0, texture),
			"mul ft0, ft0, v2" // texel color * vertex color     ft0 = surface color
		];

		if (_normalTexture != null)
		{
			fragmentShader.push(
				FilterEffect.tex("ft1", "v3", 1, normalTexture, false)
			);
			fragmentShader.push(
				"mul ft1.xy, ft1.xy, fc0.zz" // N.xy *= 2
			);
			fragmentShader.push(
				"sub ft1.xy, ft1.xy, fc0.yy" // N.xy -= 1
			);
			fragmentShader.push(
				"neg ft1.z, ft1.z" // fix direction of z axis
			);
			fragmentShader.push(
				"neg ft1.y, ft1.y" // fix direction of y axis
			);
		}
		else // use default normal vector
		{
			fragmentShader.push(
				"mov ft1, fc0.xxyy"
			);
			fragmentShader.push(
				"neg ft1.z,  ft1.z" // N = (0, 0, -1)
			);
		}

		fragmentShader.push(
			"mov ft7, v5"              // prime v5 as VECTOR_4 in FS register map (forces AGALConverter Branch 2 for m33)
		);
		fragmentShader.push(
			"m33 ft1.xyz, ft1.xyz, v5" // move N into local coords
		);
		fragmentShader.push(
			"nrm ft1.xyz, ft1.xyz"     // normalize N               ft1 = normal vector
		);

		for (i in 0...numLights)
		{
			var light:Light = _lights[i];
			var lPos:String = "fc" + (FC_LIGHT_BASE + 2 * i);
			var lCol:String = "fc" + (FC_LIGHT_BASE + 1 + 2 * i);

			if (light.type == LightSource.TYPE_AMBIENT)
			{
				fragmentShader.push(
					"mul ft2, ft0, " + lCol // illumination = surface color * ambient color
				);
				fragmentShader.push(
					"mul ft2, ft2, v4.xxxx" // illumination *= ambient ratio
				);
			}
			else
			{
				var calcLightVector:String = (light.type == LightSource.TYPE_POINT) ?
				"sub ft2, " + lPos + ", v0" :
				"mov ft2, " + lPos;

				fragmentShader.push(
					// --- calculate L . N ---
					calcLightVector
				);
				fragmentShader.push(
					nrm("ft2") // normalize light vector          ft2 = L
				);
				fragmentShader.push(
					"dp3 ft3, ft2, ft1" //                                 ft3 = L.N
				);
				fragmentShader.push(
					"sat ft3, ft3" // clamp to 0-1
				);
				fragmentShader.push(

					// --- calculate R . V ---
					"mul ft4, ft3, fc0.z" // ft4  = (L.N) * 2
				);
				fragmentShader.push(
					"mul ft4, ft4, ft1" // ft4 *= N
				);
				fragmentShader.push(
					"sub ft4, ft4, ft2" // ft4 -= L                        ft4 = R
				);
				fragmentShader.push(
					"sub ft5, fc3, v0" // calculate view vector
				);
				fragmentShader.push(
					nrm("ft5") // normalize view vector           ft5 = V
				);
				fragmentShader.push(
					"dp3 ft2, ft4, ft5" //                                 ft2 = R.V
				);
				fragmentShader.push(
					"sat ft2, ft2" // clamp to 0-1
				);
				fragmentShader.push(

					// --- calculate diffuse color ---
					"mul ft3, ft3, " + lCol // diffuse color = (L.N) * light color
				);
				fragmentShader.push(
					"mul ft3, ft3, v4.yyyy" // diffuse color *= diffuse ratio
				);
				fragmentShader.push(

					// --- calculate specular color ---
					"pow ft4, ft2, v4.wwww" // apply shininess
				);
				fragmentShader.push(
					"mul ft4, ft4, " + lCol // specular color = (R.V)^shininess * light color
				);
				fragmentShader.push(
					"mul ft4, ft4, v4.zzzz" // specular color *= specular ratio
				);
				fragmentShader.push(
					"mul ft4, ft4, ft0.wwww" // pre-multiply alpha
				);
				fragmentShader.push(

					// --- calculate total illumination from this light ---
					"mul ft2, ft0, ft3" // illumination = surface color * diffuse color
				);
				fragmentShader.push(
					"add ft2, ft2, ft4" // illumination += specular color
				);
			}

			fragmentShader.push(
				(i == 0) ? "mov ft6, ft2" :
				"add ft6, ft6, ft2" // final color += illumination
			);
		}

		if (numLights == 0)
		{
			fragmentShader.push("mov ft6, fc0.xxxx");
		}

		fragmentShader.push(
			"mov ft6.w, ft0.w" // restore alpha
		);
		fragmentShader.push(
			"mov oc, ft6"
		);

		return Program.fromSource(
			buildVertexShaderAGAL(),
			fragmentShader.join("\n"),
			AGAL_VERSION
		);
	}

	/** The vertex shader is the same on both paths.
	 *  Returns the AGAL source as a newline-joined string. */
	private function buildVertexShaderAGAL():String
	{
		var vertexShader:Array<String> = [
			"mov vt0, va4", // restore actual shininess value ...
			"mul vt0.w, vt0.w, vc5.w", // ... by multiplying with 'MAX_SHININESS'

			"m44  op, va0, vc0", // transform vertex position into clip space
			"mov  v0, va0     ", // pass vertex position to FB
			"mov  v1, va1     ", // pass texture coordinates to FP
			"mul  v2, va2, vc4", // pass vertex color * vertex alpha to FP
			"mov  v3, va3     ", // pass normal texture coordinates to FP
			"mov  v4, vt0     ", // pass material to FP

			"crs vt1.xyz, va5.xyz, va6.xyz", // calculate local z-axis
			"mul vt1.xyz, vt1.xyz, va7.xxx", // (possibly) flip local z-axis

			"mov v5.xw, va5.xw", // vertices va5, va6, vt1 contain the basis vectors of the
			"mov v6.xw, va5.yw", // local coordinate system. By storing them transposed in
			"mov v7.xw, va5.zw", // the matrix v5-v7, we'll be able to do a simple matrix
			"mov v5.y, va6.x", // transform in the fragment shader to get the
			"mov v6.y, va6.y", // normal vectors into the local coordinate system.
			"mov v7.y, va6.z",
			"mov v5.z, vt1.x",
			"mov v6.z, vt1.y",
			"mov v7.z, vt1.z"
		];

		return vertexShader.join("\n");
	}

	override private function beforeDraw(context:Context3D):Void
	{
		super.beforeDraw(context);

		// vc0-vc3 - MVP matrix
		// vc4 - alpha value (same value for all components)
		// vc5 - max shininess

		// fc0 - [0, 1, 2, 0.1]
		// fc3 - camera position

		// fc10 - light 0, position
		// fc11 - light 0, color
		// fc12 - light 1, position
		// fc13 - light 1, color
		// ...

		// va0 — vertex position (xy)
		// va1 — texture coordinates
		// va2 — vertex color (rgba), using premultiplied alpha
		// va3 - normal texture coordinates
		// va4 - material (ambientRatio, diffuseRatio, specularRatio, shininess)
		// va5 - x-axis vector (xy)
		// va6 - y-axis vector (xy)
		// va7 - z-axis scale (x) - either '1' or '-1', to flip the z-axis if necessary

		// fs0 — texture
		// fs1 - normal texture

		sVector[0] = sVector[1] = sVector[2] = sVector[3] = LightStyle.MAX_SHININESS;
		context.setProgramConstantsFromVector(Context3DProgramType.VERTEX, 5, sVector);

		sVector[0] = 0.0; sVector[1] = 1.0; sVector[2] = 2.0; sVector[3] = 0.1;
		context.setProgramConstantsFromVector(Context3DProgramType.FRAGMENT, 0, sVector);

		sVector[0] = _cameraPosition.x; sVector[1] = _cameraPosition.y;
		sVector[2] = _cameraPosition.z; sVector[3] = _cameraPosition.w;
		context.setProgramConstantsFromVector(Context3DProgramType.FRAGMENT, 3, sVector);

		// Clamp upload to the same ceiling the shader uses, so we never try to
		// write past the highest declared `fcN` uniform.
		var hardLimit:Int = getHardLimit();
		var uploadCount:Int = _lights.length < hardLimit ? _lights.length : hardLimit;

		for (i in 0...uploadCount)
		{
			var light:Light = _lights[i];

			sVector[0] = light.x; sVector[1] = light.y; sVector[2] = light.z; sVector[3] = 1.0;
			context.setProgramConstantsFromVector(
				Context3DProgramType.FRAGMENT, FC_LIGHT_BASE + 2 * i, sVector);

			Color.toVector(light.color, sVector);
			context.setProgramConstantsFromVector(
				Context3DProgramType.FRAGMENT, FC_LIGHT_BASE + 1 + 2 * i, sVector);
		}

		if (_normalTexture != null)
		{
			var repeat:Bool = textureRepeat && _normalTexture.root.isPotTexture;
			RenderUtil.setSamplerStateAt(1, _normalTexture.mipMapping, textureSmoothing, repeat);
			context.setTextureAt(1, _normalTexture.base);
		}

		vertexFormat.setVertexBufferAt(3, vertexBuffer, "normalTexCoords");
		vertexFormat.setVertexBufferAt(4, vertexBuffer, "material");
		vertexFormat.setVertexBufferAt(5, vertexBuffer, "xAxis");
		vertexFormat.setVertexBufferAt(6, vertexBuffer, "yAxis");
		vertexFormat.setVertexBufferAt(7, vertexBuffer, "zScale");
	}

	override private function afterDraw(context:Context3D):Void
	{
		context.setTextureAt(1, null);
		context.setVertexBufferAt(3, null);
		context.setVertexBufferAt(4, null);
		context.setVertexBufferAt(5, null);
		context.setVertexBufferAt(6, null);
		context.setVertexBufferAt(7, null);

		super.afterDraw(context);
	}

	/** The program-variant key uses a dual-schema encoding to keep the 32-bit
	 *  `UInt` slot collision-free for arbitrary light counts:
	 *
	 *  - **Schema A** (`numLights ≤ 8`): bit-perfect with the legacy encoding —
	 *    `lightBits << 16`, with bit 15 = 0 as the schema tag.
	 *  - **Schema B** (`numLights ≥ 9`): bit 15 = 1 (the "reserved" slot in
	 *    the legacy layout), bits 16..22 = numLights (7-bit), bits 23..31 =
	 *    9-bit FNV-1a hash of the per-light type sequence.
	 *
	 *  Disjoint by bit 15, so existing programs keep their old keys exactly. */
	override private function get_programVariantName():UInt
	{
		var normalMapBits:UInt = RenderUtil.getTextureVariantBits(_normalTexture);
		var hardLimit:Int = getHardLimit();
		var rawCount:Int = _lights.length;
		var numLights:Int = rawCount < hardLimit ? rawCount : hardLimit;
		var lightField:UInt = 0;

		if (numLights <= 8)
		{
			// ── Schema A: bit-perfect with pre-uncap builds. ────────────────
			var lightBits:UInt = 0;
			for (i in 0...numLights)
			{
				var t:UInt;
				switch _lights[i].type
				{
					case LightSource.TYPE_AMBIENT:     t = 3;
					case LightSource.TYPE_DIRECTIONAL: t = 2;
					default:                           t = 1;
				}
				lightBits = lightBits | (t << (i * 2));
			}
			lightField = lightBits << 16; // bit 15 stays 0 ⇒ Schema A
		}
		else
		{
			// ── Schema B: 7-bit count + 9-bit FNV-1a hash of types. ─────────
			var h:UInt = 0x811c9dc5; // FNV-1a 32-bit offset basis
			for (i in 0...numLights)
			{
				var t:UInt;
				switch _lights[i].type
				{
					case LightSource.TYPE_AMBIENT:     t = 3;
					case LightSource.TYPE_DIRECTIONAL: t = 2;
					default:                           t = 1;
				}
				h = (h ^ t) * 0x01000193;
			}
			var hash9:UInt = (h ^ (h >> 9) ^ (h >> 18)) & 0x1ff;
			lightField = (1 << 15)                            // Schema-B tag
					   | (((numLights : UInt) & 0x7f) << 16)  // 7-bit count
					   | (hash9 << 23);                       // 9-bit hash
		}

		return super.programVariantName | (normalMapBits << 8) | lightField;
	}

	override private function get_vertexFormat():VertexDataFormat
	{
		return VERTEX_FORMAT;
	}

	private function get_numLights():Int
	{
		return _lights.length;
	}

	private function set_numLights(value:Int):Int
	{
		var oldNumLights:Int = _lights.length;
		for (i in oldNumLights...value)
		{
			_lights[i] = new Light();
		}
		_lights.resize(value);
		return value;
	}

	public function setLightAt(index:Int, type:String, color:UInt,
							   positionOrDirection:Vector3D):Void
	{
		if (index >= numLights)
		{
			numLights = index + 1;
		}

		var light:Light = _lights[index];
		light.type = type;
		light.color = color;
		light.x = positionOrDirection.x;
		light.y = positionOrDirection.y;
		light.z = positionOrDirection.z;
	}

	/** The position of the camere in the local coordinate system of the rendered object. */
	private function get_cameraPosition():Vector3D
	{
		return _cameraPosition;
	}

	private function set_cameraPosition(value:Vector3D):Vector3D
	{
		_cameraPosition.copyFrom(value);
		return value;
	}

	private function get_normalTexture():Texture
	{
		return _normalTexture;
	}

	private function set_normalTexture(value:Texture):Texture
	{
		_normalTexture = value;
		return value;
	}
}

@:keep
class Light
{
	public var x:Float;
	public var y:Float;
	public var z:Float;
	public var color:UInt;
	public var type:String;

	public function new(color:UInt = 0xffffff, type:String = "point")
	{
		x = y = z = 0.0;
		this.color = color;
		this.type = type;
	}
}
