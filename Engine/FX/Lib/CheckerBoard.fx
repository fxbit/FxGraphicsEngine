#define VARIABLES_CHECKER(_NAME) \
	int _NAME##_CheckerX;\
	int _NAME##_CheckerY

////////////////////////////////////////////////////////////////

inline  float CheckerBoard(int PositionX,int PositionY,float Result1,float Result2){
	return ((PositionX+PositionY) % 2 == 0) ?  Result1 : Result2;
}

inline  float2 CheckerBoard(int PositionX,int PositionY,float2 Result1,float2 Result2){
	return ((PositionX+PositionY) % 2 == 0) ?  Result1 : Result2;
}

inline  float3 CheckerBoard(int PositionX,int PositionY,float3 Result1,float3 Result2){
	return ((PositionX+PositionY) % 2 == 0) ?  Result1 : Result2;
}