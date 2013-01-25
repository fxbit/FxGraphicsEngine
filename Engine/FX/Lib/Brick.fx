#define VARIABLES_BRICK(_NAME) \
	float _NAME##_BrickWidth;\
	float _NAME##_BrickHeight;\
	float _NAME##_BrickShift;\
	float _NAME##_MortarThickness

////////////////////////////////////////////////////////////////
#define BRICK(_NAME,_POSITION,_NOISE) \
Brick(_POSITION, \
_NAME##_BrickWidth, \
_NAME##_BrickHeight,\
_NAME##_BrickShift, \
_NOISE,\
_NAME##_MortarThickness)

////////////////////////////////////////////////////////////////

float HermiteCurve(float t) {return t*t*(3-2*t);} // 3t²-2t³ Called "Ease" in some art packages 
float Frac(float u) { return u - floor(u);}

// returns 0 when x<a, rising to 1 at b with a Hermite curve, holding at 1 thereafter.
float HermiteStep(float x, float a, float b) {
	if(x<a) return 0;
	if(b<x) return 1;
	return HermiteCurve((x-a)/(b-a)); // 3t^2-2t^3
}

float Brick(float2 possition, float BrickWidth, float BrickHeight,float BrickShift, float Randomness , float MortarThickness) {

	float2 InvBM = float2(1/(BrickWidth + MortarThickness + Randomness*0.04) ,
						  1/(BrickHeight+ MortarThickness + Randomness*0.04));
	float2 Border = MortarThickness*InvBM + Randomness;
	float2 vu = possition * InvBM;

	float Row=floor(vu.x);
	vu.x -= Row;
	vu.y = Frac(vu.y+(Row % 2 ? BrickShift + Randomness : 0));

	if( Border.x == 0)
		return 0;
	
	if( Border.y == 0)
		return 0;
	
	float w,h;

	w=HermiteStep(vu.x, 0, Border.x ) - HermiteStep(vu.x,  1-Border.x , 1);
	h=HermiteStep(vu.y, 0, Border.y ) - HermiteStep(vu.y,  1-Border.y, 1);
	
	return w * h;
}
