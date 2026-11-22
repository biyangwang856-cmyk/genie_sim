"""pen_cap_attractor —— 笔盖套上笔尖后把两者焊成一个整体。

挂载方式(与 benchmark_plug_017/phone_plug_attractor.py 同构):
    在 cap 资产 `Aligned.usda` 的 `entity` 上 author
        uniform asset[] omni:scripting:scripts = [@./pen_cap_attractor.py@]
    脚本文件放在该资产目录下(相对路径按 authoring layer 解析)。
    `omni.kit.scripting` 扩展在采集服务端是被加载的(server 日志里有
    `ext: omni.kit.scripting-107.3.2 startup`),不需要业务侧显式 enable。

与参考脚本的三处**必要**差异 —— 不是风格改写:

1. **笔上没有 `hole` marker prim。** 参考脚本靠 `entity/body/hole` 子 prim 定位手机
   充电孔;pen 资产只有 body/visual + 3 个 collider,没有任何 marker。所以目标点
   改成从**笔的 entity 局部系常量**算(`PEN_LOCAL_TIP`),量法见下面的几何注释。

2. **刚体挂在 `entity`,不是 `body`。** plug_017 的 `/World/body` 自带
   `PhysicsRigidBodyAPI`(实测 appliedSchemas 含它),所以参考脚本可以直接把
   `self.prim` 当刚体去建 joint。pen/cap 的 `PhysicsRigidBodyAPI` 在 `/World`
   (composed 后即 `entity`),`body` 上什么都没有。这里统一走
   `_find_rigid_ancestor()` 解析,两种资产结构都对。

3. **加了长轴平行硬判。** 参考脚本只判两个锚点是否重合。纯位置判定会在
   "笔盖横着、锚点碰巧落在笔尖上"时误吸附 —— 判定必须声明它期望什么,而不是
   只测一个代用量。

几何(entity 局部系 = payload 原生系,长轴 X;12 支笔与 12 个盖各自逐个资产完全一致,
已逐资产核对 collider 包围盒):

    笔   笔尖端点      x = -0.07802
         笔尖细段      x ∈ [-0.07802, -0.05466]   半径 4.51mm   (长 23.4mm)
         过渡段        x ∈ [-0.05466, -0.04625]   半径 8.11mm
         笔身          x ∈ [-0.04650, +0.07802]   半径 8.77mm

    盖   开口(+X 端)   x = +0.01934
         内腔          半径 8.35mm,深 38.2mm(内底面 x = -0.01887)

    ⚠ 笔身半径 8.77mm > 内腔半径 8.35mm ⇒ **笔身物理上进不去**,单侧多 0.42mm。
    过渡段名义上还差 0.24mm 能进,但两者都是**六棱**截面(外接 9.36 / 9.64),
    滚转差 ≥13° 棱角就顶平面,而没有任何环节控制这个滚转 ⇒ 实际也进不去
    (ep16 实证:命令 30mm 只前进 2.4mm)。
    所以标称插入深度取 **23.36mm = 笔尖细段长度**:细段刚好完全被盖住、过渡段齐平
    于盖口 —— 既是"盖到底"的样子,也正好是物理天花板。
    笔尖离内底面还有 15mm,不会顶死。

坐标系约定:pen 与 cap 的 local +X 在盖合状态下**同向**(盖的开口朝 +X_cap,笔尖朝
-X_pen,盖从笔尖那一侧套进去 ⇒ 两个 +X 指向同一边)。所以平行判据取
`dot(axis_cap, axis_pen) >= cos(15°)`,是 +1 附近而不是 -1 附近。
"""

from omni.kit.scripting import BehaviorScript
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from geniesim_assets.behavior_lifecycle import SceneOwner


class PenCapAttractor(BehaviorScript):
    def on_init(self):
        # 盖开口 +0.01934,标称插入 23.36mm ⇒ 笔尖落在盖局部 x = 0.01934 - 0.02336
        # 笔的**碰撞**几何是三段阶梯长方体(collider 都是 boundingCube;
        # visual 的 collisionEnabled=False,不参与碰撞):
        #     笔尖段 0~23.36mm 半宽 4.51 → 进 8.35 方孔余量 3.84mm(松套,可达)
        #     过渡段 23.36~31.52mm 半宽 8.11 → 余量 0.24mm(**压配,推不进去**)
        #     笔身   31.52mm 之后 半宽 8.77 → 进不去(物理停止点)
        #
        # ⚠ 2026-09-01 定为 0.02336 = **笔尖细段长度**,即"细段刚好完全被盖住"
        #   (用户要求"插到底")。它同时也是物理天花板:再深要让过渡段(内切 8.11)
        #   进内腔(内切 8.35),而两者都是六棱截面(外接 9.36 / 9.64),
        #   滚转差 ≥13° 就棱角顶平面 —— 没有任何环节控制这个滚转 ⇒ 进不去。
        # ⚠ 2026-08-31 我曾把这个值提到 0.028 去追过渡段的"紧配",
        #   结果 ep16 一次都没吸附:gate 最接近时 depth=22.1mm、轴向差 5.9mm,
        #   而窗口是 ±3.5mm。历史实到深度 13.3/20.5/21.6/21.8mm,从没超过 22mm ——
        #   平行爪沿杆轴只有摩擦力,推不动 0.24mm 的压配。
        #   先回退到 0.022(ep12 实测达到并焊住过),09-01 定到 0.02336(见上)。
        # ⚠ 必须与 data_gen.py 的 NOMINAL_INSERT_DEPTH 逐位一致(那边有断言核对)。
        self.nominal_insert_depth = 0.03152
        self.cap_mouth_x = 0.02127
        self.cap_local_point = Gf.Vec3d(
            self.cap_mouth_x - self.nominal_insert_depth, 0.0, 0.0
        )
        # 笔尖端点(笔 entity 局部系)
        self.pen_local_tip = Gf.Vec3d(-0.07802, 0.0, 0.0)
        # 捕获阈值。x(轴向)是固定窗口;y/z(径向)**不是固定值** —— 见下。
        # 轴向 ±6mm ⇒ 覆盖 17.4~29.4mm。这个窗口必须**宽到容纳实际能推到的深度**:
        # 历史实到 13.3/20.5/21.6/21.8mm,窗口收窄到 ±3.5mm 就把 20.5~21.8 那一段
        # 全排除了(第七轮三条全灭的直接原因)。
        # 判据宁可宽一点也不能把物理上唯一可达的区间排除在外。
        self.thresh_axial = 0.004
        # 径向阈值按**实测深度**算,不能写死:碰撞轮廓是阶梯,余量在 23.36mm 处
        # 从 3.84mm 断崖跌到 0.24mm。写死任一档都会在另一档判错 —— 取紧了浅插永远
        # 进不来,取松了深插时笔尖没对正也会误触发。
        # 这是"约束有害量本身,而不是一个代用量"的直接应用。
        self._bore_r = 0.00919          # 盖内腔方孔内半宽(×1.10)
        self._tip_half_w = 0.00451      # 笔尖段碰撞半宽
        self._trans_half_w = 0.00811    # 过渡段碰撞半宽
        self._body_half_w = 0.00877     # 笔身碰撞半宽
        self._trans_start = 0.02336     # 阶梯边界:笔尖段 → 过渡段
        self._trans_end = 0.03152       # 阶梯边界:过渡段 → 笔身(物理极限)
        # 径向阈值的下限:再紧就落进数值噪声里了
        self._radial_floor = 0.0005
        # 长轴平行度下限(见 docstring:同向,所以是 +cos)
        self.axis_align_min = 0.966  # cos(15°)

        self.attached = False
        self.joint_path = None
        self._missing_pen_warned = False
        self._last_report_time = None
        self._report_period = 2.0
        # 缓存 SingleRigidPrim view per prim_path,避免每帧重建
        self._rigid_view_cache = {}
        self._paused = False
        self._scene_owner = SceneOwner(self.prim, self._retire)

    def _retire(self):
        if self._scene_owner.valid:
            self._detach(reason="destroy")
        self.attached = False
        self.joint_path = None
        self._rigid_view_cache.clear()

    def on_destroy(self):
        self._scene_owner.close()

    def on_pause(self):
        self._paused = True


    # ------------------------------------------------------------------
    # world-xform helpers —— GPU/PhysX-aware
    # ------------------------------------------------------------------
    # 这一整节连同注释照搬 phone_plug_attractor.py —— 它记录的是踩过的坑,不要
    # 凭直觉重写。GPU pipeline (Isaac Sim `suppressReadback=False`) 下 PhysX 把
    # rigid body 位姿写到 GPU tensor buffer, USD 上 rigid body 自己的 `xformOp`
    # **不同步** (冻在 spawn), 一些版本连 Fabric usdrt attr 也是 stale。唯一稳定
    # 拿到当前物理态的路径是 IsaacSim
    # `SingleRigidPrim._rigid_prim_view.get_world_poses()` (直接读 PhysX tensor
    # buffer)。
    #
    # 修法: rigid body 走 tensor view; rigid body 的静态子孙用 tensor view
    # 读祖先 world + USD 读祖先→自己的静态 local, compose。CPU pipeline 或
    # tensor view import 失败时 fallback 到 USD 老路径 (CPU 下 USD 会自动跟
    # PhysX 同步)。

    def _rigid_view_world_pose(self, prim_path):
        """通过 IsaacSim SingleRigidPrim tensor view 拿 rigid body 当前 world pose。
        返回 Gf.Matrix4d 或 None(prim 不是 rigid, 或 API import 失败)。"""
        try:
            rp = self._rigid_view_cache.get(prim_path)
            if rp is None:
                from isaacsim.core.prims import SingleRigidPrim  # lazy import
                rp = SingleRigidPrim(prim_path=prim_path)
                self._rigid_view_cache[prim_path] = rp
            poss, oris = rp._rigid_prim_view.get_world_poses(usd=False)
            # tensor -> numpy
            try:
                p = poss[0].cpu().numpy()
                q = oris[0].cpu().numpy()
            except AttributeError:
                p = poss[0]
                q = oris[0]
            m = Gf.Matrix4d(1)
            m.SetRotate(Gf.Quatd(float(q[0]),
                                 Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))
            m.SetTranslateOnly(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            return m
        except Exception:
            return None

    def _find_rigid_ancestor(self, prim):
        """向上找最近带 RigidBodyAPI 的祖先 (可能是 prim 自己)。"""
        walker = prim
        while walker and walker.IsValid() and not walker.IsPseudoRoot():
            if walker.HasAPI(UsdPhysics.RigidBodyAPI):
                return walker
            walker = walker.GetParent()
        return None

    def _world_xform(self, prim):
        """
        当前世界矩阵。GPU pipeline 下 rigid body 走 SingleRigidPrim tensor view
        (读 PhysX GPU buffer, 实时), 静态子孙用 rigid ancestor tensor world @
        (inv(anc_static) * target_static) 组合。CPU/tensor 不可用时 fallback USD。
        """
        rigid_prim = self._find_rigid_ancestor(prim)
        if rigid_prim is not None:
            anc_world = self._rigid_view_world_pose(str(rigid_prim.GetPath()))
            if anc_world is not None:
                if rigid_prim.GetPath() == prim.GetPath():
                    return anc_world
                # Gf.Matrix4d row-vector: v_w = v_l * M。
                # target_static: prim-local origin → prim-frozen-world (USD spawn)
                # anc_static:   body-local origin → body-frozen-world (USD spawn)
                # delta = target_static * inv(anc_static): prim-local → body-local
                # composed = delta * anc_world: prim-local → body-current-world
                # (**顺序不能反**: anc_static.GetInverse() * target_static 是 bug,
                #  把世界差看成 body-local, body 旋转后 compose 结果错。)
                target_static = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
                anc_static = UsdGeom.Xformable(rigid_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default())
                delta = target_static * anc_static.GetInverse()
                composed = delta * anc_world
                return composed
        # CPU pipeline 或 tensor view 不可用 —— 老路径
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    # ------------------------------------------------------------------
    # partner lookup
    # ------------------------------------------------------------------

    def _cap_instance(self):
        """从 self.prim 往上找场景里的 cap 实例 prim。

        生成器把 oid 尾号换成随机 hash(`benchmark_cap_000` → `benchmark_cap_4eef79d3`),
        所以只能按**前缀**匹配,不能按完整 oid。
        """
        prim = self.prim
        while prim and prim.IsValid():
            if prim.GetName().startswith("benchmark_cap_"):
                return prim
            prim = prim.GetParent()
        return Usd.Prim()

    def _axis_of(self, world_mat):
        """该物体 local +X 在世界系里的单位方向。"""
        axis = world_mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        n = axis.GetLength()
        if n <= 1e-9:
            return Gf.Vec3d(0.0, 0.0, 0.0)
        return axis / n

    def _find_nearest_pen(self, cap_w):
        """在同级 prim 里找最近的笔,返回 (pen_prim, pen_world, tip_world)。"""
        cap_instance = self._cap_instance()
        if not cap_instance.IsValid():
            return Usd.Prim(), None, None

        stage = self.prim.GetStage()
        cap_anchor_world = cap_w.Transform(self.cap_local_point)
        candidates = []

        for sibling in cap_instance.GetParent().GetChildren():
            name = sibling.GetName()
            if not name.startswith("benchmark_pen_"):
                continue
            # 笔筒类资产同样以 benchmark_pen_ 开头,排掉(本任务场景里没有,
            # 但保持与参考脚本同样的防御,场景加物体时不会静默认错目标)
            if name.startswith("benchmark_pen_holder_"):
                continue

            pen = stage.GetPrimAtPath(sibling.GetPath().AppendPath("entity"))
            if not pen.IsValid():
                continue

            pen_w = self._world_xform(pen)
            tip_world = pen_w.Transform(self.pen_local_tip)
            delta = tip_world - cap_anchor_world
            # 距离相同时按 path 排序,保证多支笔时选择是确定的
            candidates.append(
                (delta.GetLength(), str(pen.GetPath()), pen, pen_w, tip_world))

        if not candidates:
            return Usd.Prim(), None, None

        _, _, pen, pen_w, tip_world = min(
            candidates, key=lambda value: (value[0], value[1]))
        return pen, pen_w, tip_world

    # ------------------------------------------------------------------
    # gate
    # ------------------------------------------------------------------

    def _pen_radius_at(self, depth):
        """插入深度 depth 时卡在盖口那个截面的**碰撞半宽**。阶梯函数,不是插值。

        碰撞几何是三个 boundingCube ⇒ 每段等宽,段间是断崖。
        """
        # 边界用 <=:深度恰好等于边界时过渡段只是齐平于盖口、还没进腔,
        # 腔内约束笔尖的仍是细段。写成 < 会让标称深度处的径向门限
        # 从 3.84mm 突降到 0.24mm,达标的 episode 也会被判死。
        if depth <= self._trans_start:
            return self._tip_half_w
        if depth < self._trans_end:
            return self._trans_half_w
        return self._body_half_w

    def _radial_thresh_at(self, depth):
        """允许的径向偏差 = **笔尖**在等宽内腔里的余量 = 9.19 − 4.51 = 4.68mm。

        与深度无关:被测量的是笔尖相对孔轴的偏移,而笔尖始终坐在那段等宽方孔里。
        ⚠ 2026-09-01 修:原来按深度切到"过渡段档"(0.24mm)。那是"过渡段能不能
          继续前进"的间隙,拿去卡"笔尖偏移"的读数属于参照系错配 —— 实测后果是
          ep3 深度 23.5mm、细段覆盖 100%(正是目标状态)却因门限塌到 0.5mm 而
          判不吸附。达成目标反而判失败,说明门限错了。
        """
        return self._bore_r - self._tip_half_w

    def _diff_in_cap_local(self, cap_w, tip_world):
        """笔尖相对盖锚点的偏差,表达在**盖局部轴**上(x 轴向 / y,z 径向)。"""
        tip_in_cap = cap_w.GetInverse().Transform(tip_world)
        return tip_in_cap - self.cap_local_point

    def _within_thresh(self, diff):
        if abs(diff[0]) >= self.thresh_axial:
            return False
        # 实测深度 = 标称 − 轴向偏差(diff[0] 为正表示笔尖还没插到锚点那么深)
        depth = self.nominal_insert_depth - diff[0]
        rt = self._radial_thresh_at(depth)
        return abs(diff[1]) < rt and abs(diff[2]) < rt

    # ------------------------------------------------------------------
    # attach / detach
    # ------------------------------------------------------------------

    def _attach(self, pen_prim, cap_w, pen_w, diff, tip_world, axis_dot):
        stage = self.prim.GetStage()

        # pen/cap 的刚体在 entity 上(不是 body),所以必须解析出真正的刚体 prim 再
        # 建 joint —— 拿 self.prim 当刚体只在 plug_017 那种结构下才成立。
        cap_body = self._find_rigid_ancestor(self.prim) or self.prim
        pen_body = self._find_rigid_ancestor(pen_prim) or pen_prim

        cap_path = cap_body.GetPath()
        pen_path = pen_body.GetPath()
        joint_path = cap_path.AppendChild("PenAttachJoint")

        # rel: cap-local → pen-local (Gf 行向量约定,顺序不能反)
        rel = cap_w * pen_w.GetInverse()
        rel_q = Gf.Quatf(rel.ExtractRotationQuat())

        existing = stage.GetPrimAtPath(joint_path)
        if existing.IsValid():
            joint = UsdPhysics.Joint(existing)
        else:
            joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

        joint.GetBody0Rel().SetTargets([cap_path])
        joint.GetBody1Rel().SetTargets([pen_path])
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(self.cap_local_point))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(self.pen_local_tip))
        joint.GetLocalRot1Attr().Set(rel_q)
        joint.GetJointEnabledAttr().Set(True)

        rb = UsdPhysics.RigidBodyAPI(pen_body)
        if rb:
            rb.GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        self.attached = True
        self.joint_path = joint_path

        anchor_world = cap_w.Transform(self.cap_local_point)
        print("[PenCapAttractor] ATTACH triggered")
        print("  joint path        : {0}".format(joint_path))
        print("  cap body          : {0}".format(cap_path))
        print("  pen body          : {0}".format(pen_path))
        print("  cap anchor (world): ({0:.5f}, {1:.5f}, {2:.5f})".format(
            anchor_world[0], anchor_world[1], anchor_world[2]))
        print("  pen tip    (world): ({0:.5f}, {1:.5f}, {2:.5f})".format(
            tip_world[0], tip_world[1], tip_world[2]))
        print("  diff in cap local : ({0:+.5f}, {1:+.5f}, {2:+.5f})".format(
            diff[0], diff[1], diff[2]))
        seat_depth = self.nominal_insert_depth - diff[0]
        print("  seated depth      : {0:.5f} m (nominal {1:.5f})".format(
            seat_depth, self.nominal_insert_depth))
        print("  threshold         : axial <{0}, radial <{1:.5f} (@depth)".format(
            self.thresh_axial, self._radial_thresh_at(seat_depth)))
        print("  axis dot          : {0:.4f} (min {1})".format(
            axis_dot, self.axis_align_min))

    def _detach(self, reason="manual"):
        if not self.attached or self.joint_path is None:
            self.attached = False
            return
        if not self.prim or not self.prim.IsValid():
            self.attached = False
            self.joint_path = None
            return
        stage = self.prim.GetStage()
        jp = stage.GetPrimAtPath(self.joint_path)
        if jp.IsValid():
            UsdPhysics.Joint(jp).GetJointEnabledAttr().Set(False)
        self.attached = False
        print("[PenCapAttractor] DETACH ({0})".format(reason))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def on_play(self):
        if not self._scene_owner.current:
            return
        if self._paused:
            self._paused = False
            return
        self._detach(reason="on_play")

    def on_stop(self):
        self._paused = False
        if not self._scene_owner.current:
            return
        self._detach(reason="on_stop")

    def on_update(self, current_time, delta_time):
        if not self._scene_owner.current:
            return
        # Reloading /World invalidates the old script's prim before Kit destroys
        # the BehaviorScript instance; one final update may still be dispatched.
        if not self.prim or not self.prim.IsValid():
            return

        # 吸附后永久固定:一旦 attach 就不再检测、不再因拉开而 detach。
        if self.attached:
            return

        cap_w = self._world_xform(self.prim)
        pen, pen_w, tip_world = self._find_nearest_pen(cap_w)
        if not pen.IsValid():
            if not self._missing_pen_warned:
                print("[PenCapAttractor] waiting: no sibling benchmark_pen_* "
                      "with an `entity` child")
                self._missing_pen_warned = True
            return
        self._missing_pen_warned = False

        diff = self._diff_in_cap_local(cap_w, tip_world)
        axis_dot = self._axis_of(cap_w) * self._axis_of(pen_w)

        # 节流诊断:不吸附时也要能看出差多少,否则失败时无从判断是姿态没对齐
        # 还是差着距离。每 _report_period 秒一行,量级可控。
        if (self._last_report_time is None
                or current_time - self._last_report_time >= self._report_period):
            self._last_report_time = current_time
            seat_depth = self.nominal_insert_depth - diff[0]
            print("[PenCapAttractor] gate diff=({0:+.4f}, {1:+.4f}, {2:+.4f}) "
                  "depth={3:+.4f} thresh=(ax {4}, rad {5:.4f}) "
                  "axis_dot={6:+.4f} (min {7})".format(
                      diff[0], diff[1], diff[2], seat_depth,
                      self.thresh_axial, self._radial_thresh_at(seat_depth),
                      axis_dot, self.axis_align_min))

        if axis_dot < self.axis_align_min:
            return
        if self._within_thresh(diff):
            self._attach(pen, cap_w, pen_w, diff, tip_world, axis_dot)
