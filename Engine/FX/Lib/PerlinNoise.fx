#define VARIABLES_PERLIN_NOISE(_NAME) \
	float _NAME##_MinValue;\
	float _NAME##_MaxValue;\
	float _NAME##_Tile_X;\
	float _NAME##_Tile_Y;\
	int _NAME##_Loops
	
////////////////////////////////////////////////////////////////
#define PERLIN_NOISE(_NAME,_POSITION,_OUTPUT) \
		_OUTPUT = 0;\
		_OUTPUT += NormalizeNoise(Noise(_POSITION##.x*_NAME##_Tile_X,\
										_POSITION##.y*_NAME##_Tile_Y,\
										_NAME##_RandomTex) ,\
								  _NAME##_MaxValue,\
								  _NAME##_MinValue);
								  
////////////////////////////////////////////////////////////////
#define PERLIN_NOISE_MULTILEVEL(_NAME,_POSITION,_OUTPUT) \
{\
	float weight = 1 ;	\
	float x = _POSITION##.x * _NAME##_Tile_X;\
	float y = _POSITION##.y * _NAME##_Tile_Y;\
	_OUTPUT = 0;\
	for(float i=0;i<_NAME##_Loops;i++){\
		_OUTPUT += Noise(	x * weight,\
							y * weight,\
							_NAME##_RandomTex) / weight;\
		weight*=2;}\
	_OUTPUT = NormalizeNoise(_OUTPUT,_NAME##_MaxValue,\
									 _NAME##_MinValue);\
}

////////////////////////////////////////////////////////////////
#define PERLIN_NOISE_MULTILEVEL_ABS(_NAME,_POSITION,_OUTPUT) \
{\
	float weight = 1 ;	\
	float x = _POSITION##.x * _NAME##_Tile_X;\
	float y = _POSITION##.y * _NAME##_Tile_Y;\
	_OUTPUT = 0;\
	for(float i=0;i<_NAME##_Loops;i++){\
		_OUTPUT += abs( Noise(  x * weight,\
								y * weight,\
								_NAME##_RandomTex) / weight);\
		weight*=2;}\
	_OUTPUT = NormalizeNoise(_OUTPUT,_NAME##_MaxValue,\
									 _NAME##_MinValue);\
}
////////////////////////////////////////////////////////////////
#define PERLIN_NOISE_MULTILEVEL_MARBLE_X(_NAME,_POSITION,_OUTPUT) \
{\
	float weight = 1 ;	\
	float x = _POSITION##.x * _NAME##_Tile_X;\
	float y = _POSITION##.y * _NAME##_Tile_Y;\
	_OUTPUT = 0;\
	for(float i=0;i<_NAME##_Loops;i++){\
		_OUTPUT += abs( Noise( x * weight,\
							   y * weight,\
							  _NAME##_RandomTex) / weight );\
		weight*=2;\
	}\
	_OUTPUT = NormalizeNoise(_OUTPUT,_NAME##_MaxValue,\
									 _NAME##_MinValue);\
	_OUTPUT =  sin ( x + _OUTPUT); \
}
////////////////////////////////////////////////////////////////
#define PERLIN_NOISE_MULTILEVEL_MARBLE_Y(_NAME,_POSITION,_OUTPUT) \
{\
	float weight = 1 ;	\
	float x = _POSITION##.x * _NAME##_Tile_X;\
	float y = _POSITION##.y * _NAME##_Tile_Y;\
	_OUTPUT = 0;\
	for(float i=0;i<_NAME##_Loops;i++){\
		_OUTPUT += abs( Noise( x * weight,\
							   y * weight,\
							  _NAME##_RandomTex) / weight );\
		weight*=2;\
	}\
	_OUTPUT = NormalizeNoise(_OUTPUT,_NAME##_MaxValue,\
									 _NAME##_MinValue);\
	_OUTPUT =  sin ( y + _OUTPUT); \
}
////////////////////////////////////////////////////////////////
/// Smooth the entry value
/// <param name="t">The entry value</param>
/// <returns>The smoothed value</returns>
float Fade(float t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

/// Modifies the result by adding a directional bias
/// <param name="hash">The random value telling in which direction the bias will occur</param>
/// <param name="x">The amount of the bias on the X axis</param>
/// <param name="y">The amount of the bias on the Y axis</param>
/// <returns>The directional bias strength</returns>
float Grad(int hash, float x, float y)
{
    // Fetch the last 3 bits
    int h = hash & 3;

    // Result table for U
    // ---+------+---+------
    //  0 | 0000 | x |  x
    //  1 | 0001 | x |  x
    //  2 | 0010 | x | -x
    //  3 | 0011 | x | -x

    float u = (h & 2) == 0 ? x : -x;

    // Result table for V
    // ---+------+---+------
    //  0 | 0000 | y |  y
    //  1 | 0001 | y | -y
    //  2 | 0010 | y |  y
    //  3 | 0011 | y | -y

    float v = (h & 1) == 0 ? y : -y;

    // Result table for U + V
    // ---+------+----+----+--
    //  0 | 0000 |  x |  y |  x + y
    //  1 | 0001 |  x | -y |  x - y
    //  2 | 0010 | -x |  y | -x + y
    //  3 | 0011 | -x | -y | -x - y

    return u + v;
}
/* * /
/// Generates a bi-dimensional noise
float Noise(float x, float y, int RandTex[256])
{
	// Compute the cell coordinates
	int X = (int)floor(x) & 255;
	int Y = (int)floor(y) & 255;

	// Retrieve the decimal part of the cell
	x -= floor(x);
	y -= floor(y);

	// Smooth the curve
	float u = Fade(x);
	float v = Fade(y);

	// Fetch some random values from the table
	int A = RandTex[X] + Y;
	int B = RandTex[X + 1] + Y; 

	// Interpolate between directions 
	return lerp(lerp(Grad(RandTex[A]	, x    , y	  ) ,
					 Grad(RandTex[B]	, x - 1, y	  ) , u),
				lerp(Grad(RandTex[A+1]	, x    , y - 1) ,
					 Grad(RandTex[B+1]	, x - 1, y - 1) , u) , v);
}
/ * */
float Noise(float x, float y, Texture1D <int> RandTex)
{
	// Compute the cell coordinates
	int X = (int)floor(x) & 255;
	int Y = (int)floor(y) & 255;

	// Retrieve the decimal part of the cell
	x -= floor(x);
	y -= floor(y);

	// Smooth the curve
	float u = Fade(x);
	float v = Fade(y);

	// Fetch some random values from the table
	int A = RandTex.Load( int2(X		,0)) + Y;
	int B = RandTex.Load( int2(X + 1	,0)) + Y; 

	// Interpolate between directions 
	return lerp(lerp(Grad(RandTex.Load( int2(A		,0))	, x    , y	  ) ,
					 Grad(RandTex.Load( int2(B		,0))	, x - 1, y	  ) , u),
				lerp(Grad(RandTex.Load( int2(A + 1	,0))	, x    , y - 1) ,
					 Grad(RandTex.Load( int2(B + 1	,0))	, x - 1, y - 1) , u) , v);
}/**/

////////////////////////////////////////////////////////////////
/// Smooth the entry value
/// <param name="t">The entry value</param>
/// <returns>The smoothed value</returns>
float2 Fade(float2 t)
{
    return t * t * t * (t * (t * 6 - 15) + 10);
}

float Noise(float2 xy, Texture1D <int> RandTex)
{
	// Compute the cell coordinates
	int2 XY = (int2)floor(xy) & 255;

	// Retrieve the decimal part of the cell
	xy -= floor(xy);

	// Smooth the curve
	float2 uv = Fade(xy);

	// Fetch some random values from the table
	int A = RandTex.Load( int2(XY.x		,0)) + XY.y;
	int B = RandTex.Load( int2(XY.x + 1	,0)) + XY.y; 

	// Interpolate between directions 
	return lerp(lerp(Grad(RandTex.Load( int2(A		,0))	, xy.x    , xy.y	  ) ,
					 Grad(RandTex.Load( int2(B		,0))	, xy.x - 1, xy.y	  ) , uv.x),
				lerp(Grad(RandTex.Load( int2(A + 1	,0))	, xy.x    , xy.y - 1  ) ,
					 Grad(RandTex.Load( int2(B + 1	,0))	, xy.x - 1, xy.y - 1  ) , uv.x) , uv.y);
}

/// scale the values to the output
float NormalizeNoise(float value, float MaxValue,float MinValue){
	return (((value + 1) * 0.5) * (MaxValue - MinValue) + MinValue);
}
